#!/usr/bin/env python3
"""Create a content-addressed result archive without following symlinks."""
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

from cloud_common import hash_tree, sha256_file, utc_now, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"missing run directory: {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_sha, files = hash_tree(run_dir)
    manifest = {
        "schema_version": "xrd-cloud5090-result-package-v1",
        "created_at": utc_now(),
        "source_tree_sha256": tree_sha,
        "files": files,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    archive = out_dir / f"xrd-cloud5090-results-{tree_sha[:16]}.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = 0
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for item in files:
            source = run_dir / item["path"]
            tar_info = tar.gettarinfo(str(source), arcname=f"results/{item['path']}")
            tar_info.mtime = 0
            with source.open("rb") as handle:
                tar.addfile(tar_info, handle)
    receipt = {
        **manifest,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
    }
    write_json(out_dir / f"xrd-cloud5090-results-{tree_sha[:16]}.json", receipt)
    print(json.dumps({"archive": str(archive), "sha256": receipt["archive_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
