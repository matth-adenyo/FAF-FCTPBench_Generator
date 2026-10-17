"""Compact benchmark packaging utilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_benchmark(
    instance_root: str | Path,
    catalogue_path: str | Path,
    output_path: str | Path,
    *,
    screening_path: str | Path | None = None,
    characterization_directory: str | Path | None = None,
    results_directory: str | Path | None = None,
) -> Path:
    """Package Parquet-heavy benchmark data."""
    root = Path(instance_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        def add(path: Path, arcname: str) -> None:
            compression = zipfile.ZIP_STORED if path.suffix.lower() == ".parquet" else zipfile.ZIP_DEFLATED
            archive.write(path, arcname=arcname, compress_type=compression)

        catalogue = Path(catalogue_path)
        add(catalogue, f"catalogue/{catalogue.name}")
        if results_directory is not None:
            rroot = Path(results_directory)
            if rroot.is_dir():
                for path in sorted(p for p in rroot.rglob("*") if p.is_file() and "checkpoints" not in p.parts):
                    add(path, f"results/{path.relative_to(rroot).as_posix()}")
        else:
            if screening_path is not None and Path(screening_path).is_file():
                screen = Path(screening_path); add(screen, f"results/{screen.name}")
                protocol = screen.with_suffix(".protocol.json")
                if protocol.is_file(): add(protocol, f"results/{protocol.name}")
            if characterization_directory is not None:
                croot = Path(characterization_directory)
                if croot.is_dir():
                    for path in sorted(p for p in croot.rglob("*") if p.is_file()):
                        add(path, f"results/characterization/{path.relative_to(croot).as_posix()}")
        for instance_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for path in sorted(p for p in instance_dir.rglob("*") if p.is_file()):
                add(path, f"instances/{instance_dir.name}/{path.relative_to(instance_dir).as_posix()}")
    manifest = {
        "archive": output.name,
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "storage_note": "Parquet tables are Zstandard-compressed.",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return output
