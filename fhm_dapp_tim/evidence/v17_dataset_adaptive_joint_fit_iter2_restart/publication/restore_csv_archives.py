"""Verify or restore the exact CSV bytes recorded in ARCHIVE_MANIFEST.json."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true", help="restore missing CSVs; never overwrite conflicting files")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = json.loads(Path(__file__).with_name("ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["csv_files"]:
        target = root / item["original_path"]
        if not target.resolve().is_relative_to(root):
            raise ValueError("CSV path outside repository")
        exists = target.exists()
        if exists and (target.stat().st_size != item["original_bytes"] or sha256(target) != item["original_sha256"]):
            raise ValueError(f"Existing CSV has different bytes: {target}")
        output = target.open("xb") if args.restore and not exists else None
        digest = hashlib.sha256()
        size = 0
        try:
            for part in item["gzip_parts"]:
                archive = root / part["path"]
                if not archive.resolve().is_relative_to(root):
                    raise ValueError("Archive path outside repository")
                if archive.stat().st_size != part["bytes"] or sha256(archive) != part["sha256"]:
                    raise ValueError(f"Archive hash mismatch: {archive}")
                with gzip.open(archive, "rb") as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                        size += len(block)
                        if output is not None:
                            output.write(block)
        finally:
            if output is not None:
                output.close()
        if size != item["original_bytes"] or digest.hexdigest() != item["original_sha256"]:
            raise ValueError(f"Decompressed CSV hash mismatch: {target}")
        print(f"OK {item['original_path']}")
    print(f"Verified {len(manifest['csv_files'])} CSV files; restore={args.restore}")


if __name__ == "__main__":
    main()
