"""Create and verify a source-and-evidence submission archive; never publishes it."""
from pathlib import Path
import hashlib
import json
import zipfile
from check_upload import upload_files

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=upload_files()
    missing=[str(p) for p in files if not p.is_file()]
    if missing:raise FileNotFoundError(missing)
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    archive=ROOT/"output"/"HackSHSID_AIDC_Submission.zip"
    source_archive=ROOT/"output"/"AIDC_LaTeX_Source.zip"
    with zipfile.ZipFile(source_archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files:
            if (ROOT/"output"/"reports"/"latex") in p.parents:z.write(p,p.relative_to(ROOT/"output"/"reports"/"latex").as_posix())
    with zipfile.ZipFile(source_archive) as z:
        if z.testzip() is not None:raise RuntimeError("Corrupt LaTeX source archive")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        z.writestr("PACKAGE_MANIFEST.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError("Corrupt archive")
        for entry in manifest:
            if hashlib.sha256(z.read(entry["path"])).hexdigest()!=entry["sha256"]:
                raise RuntimeError("Archive checksum mismatch")
    print(f"Verified {len(files)} files: {archive} ({archive.stat().st_size:,} bytes)")


if __name__=="__main__":main()
