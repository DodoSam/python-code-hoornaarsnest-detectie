
import requests
import time
import csv
import os
import tkinter as tk
from tkinter import filedialog
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.inaturalist.org/v1"

# ------------------------------------------
# 0. SELECT DOWNLOAD FOLDER
# ------------------------------------------

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

OUT = filedialog.askdirectory(
    title="Choose where to save hornet nest images"
)

root.destroy()

if not OUT:
    print("No folder selected. Download cancelled.")
    raise SystemExit

os.makedirs(OUT, exist_ok=True)

print(f"Images will be saved to: {OUT}")

# ------------------------------------------
# 1. SEARCH TERMS
# ------------------------------------------

SEARCH_TERMS = [
    "nest",            # English / Dutch
    "nests",
    "hornet nest",
    "wasp nest",
    "hive",
    "nid",             # French
    "nid de frelon",
    "nid de frelon asiatique",
    "nido",            # Spanish / Italian
    "nido de avispa",
    "ninho",           # Portuguese
    "ninho de vespa",
    "Nest",            # German
    "Hornissennest",
    "Wespennest",      # Dutch / German
    "hoornaarsnest",  # Dutch
    "wespennest",      # Dutch
]

# Search only observation descriptions
# and tags, not taxon names.
SEARCH_FIELDS = [
    "description",
    "tags"
]

# ------------------------------------------
# 2. API SESSION WITH RETRIES
# ------------------------------------------

session = requests.Session()

session.headers.update({
    "User-Agent": "HornetNestResearchDownloader/1.0"
})

retry = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("https://", adapter)

# ------------------------------------------
# 3. FIND VESPA VELUTINA TAXON ID
# ------------------------------------------

response = session.get(
    f"{API}/taxa",
    params={
        "q": "Vespa velutina",
        "rank": "species"
    },
    timeout=30
)

response.raise_for_status()

taxa = response.json()["results"]

matches = [
    taxon for taxon in taxa
    if taxon.get("name", "").lower() == "vespa velutina"
]

if not matches:
    raise RuntimeError("Vespa velutina taxon not found.")

taxon_id = matches[0]["id"]

print("Taxon ID:", taxon_id)

# ------------------------------------------
# 4. METADATA CONFIGURATION
# ------------------------------------------

metadata_path = os.path.join(OUT, "metadata.csv")

fieldnames = [
    "bestand",
    "observatie_id",
    "datum",
    "locatie",
    "licentie",
    "fotograaf",
    "url",
    "beschrijving",
    "zoekterm"
]

rows_by_filename = {}

# Load existing metadata to support reruns.
if os.path.exists(metadata_path):
    with open(
        metadata_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            filename = row.get("bestand")
            if filename:
                rows_by_filename[filename] = row

print(
    f"Existing metadata records: "
    f"{len(rows_by_filename)}"
)

# ------------------------------------------
# 5. DOWNLOAD SETTINGS
# ------------------------------------------

ALLOWED_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-nc"
}

total_downloaded = 0
total_skipped = 0
total_failed = 0

def save_metadata():
    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        writer.writerows(
            rows_by_filename.values()
        )

def get_large_url(url):
    if not url:
        return None

    # Replace only the photo-size component.
    for size in [
        "square", "thumb", "small",
        "medium", "original"
    ]:
        url = url.replace(
            f"/{size}.",
            "/large."
        )

    return url

# ------------------------------------------
# 6. SEARCH ALL TERMS
# ------------------------------------------

for search_field in SEARCH_FIELDS:

    for term in SEARCH_TERMS:

        print("\n" + "=" * 55)
        print(f"Searching: {term}")
        print(f"Field: {search_field}")
        print("=" * 55)

        params = {
            "taxon_id": taxon_id,
            "photos": "true",
            "q": term,
            "search_on": search_field,
            "photo_license": "cc0,cc-by,cc-by-nc",
            "per_page": 200,
            "order_by": "id",
            "order": "asc",
            "id_above": 0
        }

        search_count = 0
        page_count = 0

        while True:

            try:
                response = session.get(
                    f"{API}/observations",
                    params=params,
                    timeout=60
                )

                response.raise_for_status()

                data = response.json()

            except (
                requests.RequestException,
                ValueError
            ) as e:

                print(
                    f"Search failed for '{term}' "
                    f"({search_field}): {e}"
                )
                break

            results = data.get("results", [])

            if not results:
                break

            page_count += 1

            print(
                f"Page {page_count}: "
                f"{len(results)} observations"
            )

            # ----------------------------------
            # PROCESS OBSERVATIONS
            # ----------------------------------

            for obs in results:

                obs_id = obs["id"]

                for i, photo in enumerate(
                    obs.get("photos", [])
                ):

                    license_code = (
                        photo.get("license_code") or ""
                    ).lower()

                    # Ensure individual photo
                    # has an allowed license.
                    if license_code not in ALLOWED_LICENSES:
                        continue

                    photo_url = get_large_url(
                        photo.get("url")
                    )

                    if not photo_url:
                        continue

                    filename = f"{obs_id}_{i}.jpg"

                    path = os.path.join(
                        OUT,
                        filename
                    )

                    # --------------------------------
                    # SKIP EXISTING IMAGES
                    # --------------------------------

                    if os.path.exists(path):
                        total_skipped += 1

                    else:

                        try:
                            img = session.get(
                                photo_url,
                                timeout=30
                            )

                            img.raise_for_status()

                            # Avoid saving an empty file.
                            if not img.content:
                                raise ValueError(
                                    "Empty image response"
                                )

                            # Write to a temporary file first.
                            tmp_path = path + ".part"

                            with open(
                                tmp_path,
                                "wb"
                            ) as f:
                                f.write(img.content)

                            os.replace(
                                tmp_path,
                                path
                            )

                            total_downloaded += 1
                            search_count += 1

                            print(
                                f"Downloaded: {filename}"
                            )

                            time.sleep(0.5)

                        except (
                            requests.RequestException,
                            OSError,
                            ValueError
                        ) as e:

                            total_failed += 1

                            print(
                                f"Failed: {filename}: {e}"
                            )

                            tmp_path = path + ".part"

                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

                            continue

                    # --------------------------------
                    # SAVE PHOTO METADATA
                    # --------------------------------

                    if filename not in rows_by_filename:

                        rows_by_filename[filename] = {
                            "bestand": filename,
                            "observatie_id": obs_id,
                            "datum": obs.get(
                                "observed_on"
                            ),
                            "locatie": obs.get(
                                "location"
                            ),
                            "licentie": license_code,
                            "fotograaf": photo.get(
                                "attribution"
                            ),
                            "url": obs.get("uri"),
                            "beschrijving": obs.get(
                                "description"
                            ),
                            "zoekterm": term
                        }

            # ----------------------------------
            # NEXT PAGE USING ID PAGINATION
            # ----------------------------------

            last_id = results[-1]["id"]

            if last_id <= params["id_above"]:
                print("Pagination stopped: no progress.")
                break

            params["id_above"] = last_id

            # Save progress after each page.
            save_metadata()

            # Respect API rate limits.
            time.sleep(1)

        print(
            f"Completed '{term}' ({search_field}): "
            f"{search_count} new images"
        )

        save_metadata()
        time.sleep(1)

# ------------------------------------------
# 7. SAVE FINAL METADATA
# ------------------------------------------

save_metadata()

# ------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------

print("\n" + "=" * 55)
print("DOWNLOAD COMPLETED")
print("=" * 55)

print(f"New images downloaded: {total_downloaded}")
print(f"Existing/duplicate matches: {total_skipped}")
print(f"Failed downloads: {total_failed}")
print(f"Total metadata records: {len(rows_by_filename)}")

print(f"\nFolder: {OUT}")
print(f"Metadata: {metadata_path}")

print("\nFinished!")
