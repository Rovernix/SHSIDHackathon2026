"""Build the curated GitHub snapshot, verify copy hashes, and smoke-test it."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime

from check_upload import ROOT, upload_files


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    subprocess.run([sys.executable, "-B", str(ROOT/"tools/check_upload.py")], cwd=ROOT, check=True)
    target = (ROOT/"push").resolve()
    if not target.is_relative_to(ROOT) or target == ROOT:
        raise ValueError("The upload target must remain inside this repository")
    target.mkdir(exist_ok=True)
    files = upload_files()
    relatives = {p.relative_to(ROOT).as_posix() for p in files}
    allowed_extra = {"PUSH_MANIFEST.json", "results/qa/staging_qa.json", "output/AIDC_Modeling_Report.pdf"}
    previous_path = target/"PUSH_MANIFEST.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        stale = [row for row in previous["files"] if row["path"] not in relatives | allowed_extra]
        # Retire only files whose bytes still match the prior generated snapshot.
        # Preserve all unrecognized/user-edited content, and archive before unlinking.
        retired = (ROOT/"archive"/("retired_upload_"+datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f"))).resolve()
        if not retired.is_relative_to(ROOT/"archive"):
            raise ValueError("Retirement archive escaped the workspace")
        for row in stale:
            source=(target/row["path"]).resolve()
            if not source.is_relative_to(target):
                raise ValueError("Prior manifest contains an unsafe path")
            if source.exists() and digest(source) != row["sha256"]:
                raise ValueError("Preserve modified retired snapshot file: "+row["path"])
        for row in stale:
            source=(target/row["path"]).resolve()
            if not source.exists():
                continue
            destination=(retired/row["path"]).resolve()
            if not destination.is_relative_to(retired):
                raise ValueError("Archive target escaped its directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source,destination)
            if digest(source) != digest(destination):
                raise IOError("Retirement archive copy differs")
            source.unlink()
    extras = [p.relative_to(target).as_posix() for p in target.rglob("*")
              if p.is_file() and p.relative_to(target).as_posix() not in relatives | allowed_extra
              and "tmp" not in p.relative_to(target).parts
              and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}]
    if extras:
        raise ValueError("Unrecognized files already in the upload folder: " + ", ".join(extras))
    for source in files:
        destination = target/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if digest(source) != digest(destination):
            raise IOError("Snapshot copy differs: " + str(source.relative_to(ROOT)))
    # Earlier desktop links may hold this compatibility copy open with writes
    # permitted but renaming prohibited. Keep its bytes current and git-ignore it;
    # the canonical report and manifest use output/reports/ exclusively.
    legacy = target/"output/AIDC_Modeling_Report.pdf"
    if legacy.exists():
        shutil.copyfile(ROOT/"output/reports/AIDC_Modeling_Report.pdf", legacy)
    verification = subprocess.run([sys.executable, "-B", "tools/verify_results.py"], cwd=target, check=True,
                                  capture_output=True, text=True)
    cli = subprocess.run([sys.executable, "-B", "tools/qa_cli.py"], cwd=target, check=True,
                         capture_output=True, text=True)
    # Remove only the four scratch files this CLI smoke test creates. Preserve
    # any other pre-existing content; empty scratch directories can disappear.
    scratch=(target/"tmp/cli").resolve()
    if not scratch.is_relative_to(target):
        raise ValueError("CLI scratch path escaped the upload folder")
    for name in ["tiny.json","public.json","solution.json","infeasible.json"]:
        (scratch/name).unlink(missing_ok=True)
    for folder in [scratch,scratch.parent]:
        try:folder.rmdir()
        except OSError:pass
    staging = dict(passed=True, source_files_copied=len(files), exact_copy_hashes_before_smoke_tests=True,
                   standalone_unit_and_audit=verification.stdout.strip(), standalone_cli=cli.stdout.strip(),
                   scope="Browser/PDF artifacts are copied unchanged from the checked originals; Python tests, public input rebuild and CLI checks execute again inside the snapshot.")
    (target/"results/qa/staging_qa.json").write_text(json.dumps(staging, indent=2)+"\n", encoding="utf-8")
    subprocess.run([sys.executable, "-B", "tools/check_upload.py"], cwd=target, check=True)
    print(f"Prepared standalone snapshot: {target}")
    print(verification.stdout.strip())
    print(cli.stdout.strip())


if __name__ == "__main__":
    main()
