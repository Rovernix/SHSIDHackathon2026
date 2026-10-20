"""Download/verify open sources and extract the compact, shipped raw excerpts.

Normal users can reproduce experiments offline from data/raw/. This command is
for rebuilding those excerpts from their original archives. Large downloads
stay in tmp/open_data_cache and are never included in the GitHub snapshot.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aidc.data import save_json
from aidc.paths import ROOT, RAW

HELIOS_COMMIT = "159f0caeec16600b9b6017862952a36aae01c43f"
OPSD_COMMIT = "e1d671dbddf222dbaeb80cf7c4fa8d56b25fa365"
MIRROR_COMMIT = "c5991843669f084aa6452b3fe2f979a2d33aa678"
HELIOS_URL = f"https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/{HELIOS_COMMIT}/data.zip"
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series.sqlite"
MIRROR_URL = f"https://raw.githubusercontent.com/FlorianRauls/german-power-grid-dashboard/{MIRROR_COMMIT}/data/time_series.zip"
CARBON_URL = "https://api.carbonintensity.org.uk/intensity/2020-05-31T00:00Z/2020-06-10T00:00Z"
HELIOS_SHA = "3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac"
OPSD_SHA = "37ef381572b669937712f85e8c5b40f4ef6148bf9a288424c503e2db2e72f51c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_small(url, path, offline):
    if path.is_file():
        return
    if offline:
        raise FileNotFoundError(path)
    request = urllib.request.Request(url, headers={"User-Agent": "HackSHSID-AIDC-research/2.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read()
        expected = response.headers.get("Content-Length")
        if expected is not None and len(content) != int(expected):
            raise IOError("Truncated download: " + url)
    path.write_bytes(content)


def fetch_ranges(url, path, size):
    """Bounded, resumable ranges; reject truncated or misaddressed responses."""
    chunk_size = 1024 * 1024
    part_dir = path.parent / (path.name + ".parts")
    part_dir.mkdir(exist_ok=True)

    def get(start):
        end = min(size - 1, start + chunk_size - 1)
        part = part_dir / str(start)
        if part.is_file() and part.stat().st_size == end - start + 1:
            return start, part
        last = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={
                    "User-Agent": "HackSHSID-AIDC-research/2.0", "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise IOError("Server did not honor the requested byte range")
                    content = response.read()
                if len(content) != end - start + 1:
                    raise IOError("Truncated byte range")
                part.write_bytes(content)
                return start, part
            except Exception as error:
                last = error
        raise last

    parts = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(get, start) for start in range(0, size, chunk_size)]):
            start, part = future.result()
            parts[start] = part
    with path.open("wb") as output:
        for start in sorted(parts):
            with parts[start].open("rb") as source:
                shutil.copyfileobj(source, output)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "tmp/open_data_cache")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded full archives only")
    args = parser.parse_args()
    cache = args.cache
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "helios_complete.zip"
    database = cache / "time_series.sqlite"
    if not archive.exists() or digest(archive) != HELIOS_SHA:
        if args.offline:
            raise ValueError("The verified Helios archive is absent")
        fetch_ranges(HELIOS_URL, archive, 36437672)
    if digest(archive) != HELIOS_SHA:
        raise ValueError("Helios SHA-256 mismatch")
    if not database.exists() or digest(database) != OPSD_SHA:
        if args.offline:
            raise ValueError("The verified OPSD database is absent")
        try:
            fetch_small(OPSD_URL, database, False)
            if digest(database) != OPSD_SHA:
                raise ValueError("Official OPSD download failed its checksum")
        except Exception:
            mirror = cache / "opsd_mirror.zip"
            fetch_ranges(MIRROR_URL, mirror, 90998431)
            with zipfile.ZipFile(mirror) as zipped:
                database.write_bytes(zipped.read("time_series.sqlite"))
    if digest(database) != OPSD_SHA:
        raise ValueError("OPSD SHA-256 mismatch; mirror bytes must match the original publication")

    resources = [
        ("helios_readme.md", f"https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/{HELIOS_COMMIT}/README.md", "helios/README.source.md"),
        ("helios_license.txt", f"https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/{HELIOS_COMMIT}/LICENSE.txt", "helios/LICENSE.txt"),
        ("opsd_datapackage.json", "https://data.open-power-system-data.org/time_series/2020-10-06/datapackage.json", "opsd/datapackage.json"),
        ("opsd_checksums.txt", f"https://raw.githubusercontent.com/Open-Power-System-Data/time_series/{OPSD_COMMIT}/checksums.txt", "opsd/checksums.txt"),
        ("opsd_source_readme.md", "https://data.open-power-system-data.org/time_series/2020-10-06/README.md", "opsd/README.source.md"),
        ("neso_2020-05-31_to_10.json", CARBON_URL, "neso/intensity.json"),
    ]
    for name, url, relative in resources:
        fetch_small(url, cache / name, args.offline)
        destination = RAW / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cache / name, destination)
    checksums = dict(line.split(",") for line in (RAW / "opsd/checksums.txt").read_text().splitlines())
    if checksums["time_series.sqlite"] != digest(database):
        raise ValueError("OPSD database differs from the publisher's own checksum file")

    fields = ["source_row", "job_id", "vc", "gpu_num", "cpu_num", "node_num", "state",
              "submit_time", "start_time", "end_time", "duration", "queue"]
    selected = []
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("Helios ZIP member CRC failure")
        with zipped.open("data/Saturn/cluster_log.csv") as member:
            for index, row in enumerate(csv.DictReader(io.TextIOWrapper(member)), start=2):
                if row["vc"] != "vcOIr":
                    continue
                submitted = "2020-05-31" <= row["submit_time"][:10] < "2020-06-10"
                short_overlap = (row["state"] == "COMPLETED" and 0 < int(row["duration"]) < 1800 and
                                 row["start_time"] < "2020-06-10" and row["end_time"] >= "2020-05-31")
                if submitted or short_overlap:
                    selected.append(dict(source_row=index, **{key: row[key] for key in fields if key != "source_row"}))
        write_csv(RAW / "helios/saturn_vc_jobs.csv", fields, selected)
        with zipped.open("data/Saturn/cluster_gpu_number.csv") as member:
            capacity = [dict(date=row["date"], gpu_count=row["vcOIr"], source_total_gpus=row["total"])
                        for row in csv.DictReader(io.TextIOWrapper(member))
                        if "2020-05-31" <= row["date"] < "2020-06-10"]
        write_csv(RAW / "helios/saturn_vc_capacity.csv", ["date", "gpu_count", "source_total_gpus"], capacity)

    energy_fields = ["utc_timestamp", "GB_GBN_price_day_ahead", "GB_GBN_solar_generation_actual",
                     "GB_GBN_load_actual_entsoe_transparency", "GB_GBN_load_forecast_entsoe_transparency"]
    with sqlite3.connect(database) as db:
        query = ('SELECT ' + ','.join('"' + field + '"' for field in energy_fields) +
                 ' FROM time_series_60min_singleindex WHERE utc_timestamp>=? AND utc_timestamp<? ORDER BY utc_timestamp')
        energy = [dict(zip(energy_fields, row)) for row in db.execute(query, ("2020-05-31T00:00:00Z", "2020-06-10T00:00:00Z"))]
    if len(energy) != 240 or any(row[k] is None for row in energy for k in energy_fields):
        raise ValueError("Incomplete required public electricity observations")
    write_csv(RAW / "opsd/gb_hourly.csv", energy_fields, energy)

    sources = [
        dict(id="helios", publisher="SenseTime / S-Lab System Group", canonical_url="https://github.com/S-Lab-System-Group/HeliosData",
             download_url=HELIOS_URL, commit=HELIOS_COMMIT, archive_sha256=HELIOS_SHA, license="CC BY 4.0",
             citation="Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters, SC 2021",
             extraction="Saturn / vcOIr only; selected columns and original CSV row numbers retained"),
        dict(id="opsd", publisher="Open Power System Data; original ENTSO-E Transparency records",
             canonical_url=OPSD_URL, version="2020-10-06", doi="10.25832/time_series/2020-10-06",
             archival_mirror_url=MIRROR_URL, mirror_commit=MIRROR_COMMIT,
             official_checksum_url=f"https://raw.githubusercontent.com/Open-Power-System-Data/time_series/{OPSD_COMMIT}/checksums.txt",
             database_sha256=OPSD_SHA, matches_official_database_checksum=True,
             license="Source-specific ENTSO-E terms; OPSD attribution retained, no blanket license asserted",
             extraction="240 hourly British records, selected columns, no interpolation or clipping"),
        dict(id="neso", publisher="UK Carbon Intensity API", canonical_url=CARBON_URL,
             license="CC BY 4.0", extraction="Unmodified API response; exact from/to intervals retained"),
    ]
    files = [dict(path=p.relative_to(RAW).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(RAW.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    save_json(RAW / "manifest.json", dict(sources=sources, files=files, extraction_version=1,
        selected_vc="vcOIr", helios_excerpt_rows=len(selected), opsd_excerpt_rows=len(energy),
        note="The source database was recovered from a public mirror and verified byte-for-byte with the publisher's checksum. No generated fallback observations are permitted."))
    print(f"Verified source archives; extracted {len(selected)} Helios records and {len(energy)} OPSD hours.")


if __name__ == "__main__":
    main()
