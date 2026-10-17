"""Solver-neutral, compact representation of FAF-FCTPBench instances."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd

TABLE_BASENAMES = (
    "sources", "destinations", "products", "scenarios", "lanes",
    "supplies", "demands", "variable_costs",
)
TABLE_FILES = {name: f"{name}.parquet" for name in TABLE_BASENAMES}
STORAGE_MANIFEST = "storage_manifest.json"
STRING_COLUMNS = {
    "source_id", "source_name", "destination_id", "destination_name",
    "product_id", "product_name", "scenario_id", "provenance", "lane_id",
}
FLOAT_COLUMNS = {
    "probability", "distance_miles", "stress_load_ktons", "capacity_ktons", "fixed_cost_usd",
    "base_flow_ktons", "supply_ktons", "demand_ktons", "cost_usd_per_kton",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _typed_table(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column in STRING_COLUMNS:
            result[column] = result[column].astype("string")
        elif column in FLOAT_COLUMNS:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    return result


def _compact_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"


def table_files_for_directory(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    if all((root / filename).is_file() for filename in TABLE_FILES.values()):
        return TABLE_FILES
    raise FileNotFoundError(f"{root} does not contain the complete canonical Parquet table set")


@dataclass
class FCTPInstance:
    sources: pd.DataFrame
    destinations: pd.DataFrame
    products: pd.DataFrame
    scenarios: pd.DataFrame
    lanes: pd.DataFrame
    supplies: pd.DataFrame
    demands: pd.DataFrame
    variable_costs: pd.DataFrame
    metadata: dict[str, object]

    def write(
        self,
        directory: str | Path,
        *,
        compression_level: int = 9,
        dictionary_encoding: bool = True,
    ) -> Path:
        """Write compact Apache Parquet tables with Zstandard compression."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error: 
            raise RuntimeError("writing benchmark instances requires pyarrow") from error

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        table_manifest: dict[str, dict[str, object]] = {}
        for attribute, filename in TABLE_FILES.items():
            table_frame = _typed_table(getattr(self, attribute))
            path = output / filename
            arrow = pa.Table.from_pandas(table_frame, preserve_index=False)
            pq.write_table(
                arrow,
                path,
                compression="zstd",
                compression_level=int(compression_level),
                use_dictionary=bool(dictionary_encoding),
                write_statistics=True,
            )
            table_manifest[attribute] = {
                "file": filename,
                "rows": int(len(table_frame)),
                "columns": list(table_frame.columns),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        metadata = dict(self.metadata)
        metadata["storage"] = {
            "format": "Apache Parquet",
            "compression": "Zstandard",
            "compression_level": int(compression_level),
            "dictionary_encoding": bool(dictionary_encoding),
        }
        metadata_path = output / "metadata.json"
        metadata_path.write_text(_compact_json(metadata), encoding="utf-8")
        manifest = {
            "format": "Apache Parquet",
            "compression": "Zstandard",
            "compression_level": int(compression_level),
            "dictionary_encoding": bool(dictionary_encoding),
            "tables": table_manifest,
            "metadata": {"file": metadata_path.name, "sha256": _sha256(metadata_path)},
        }
        (output / STORAGE_MANIFEST).write_text(_compact_json(manifest), encoding="utf-8")
        return output

    @classmethod
    def read(cls, directory: str | Path) -> "FCTPInstance":
        source = Path(directory)
        table_files = table_files_for_directory(source)
        tables = {
            attribute: pd.read_parquet(source / filename, engine="pyarrow")
            for attribute, filename in table_files.items()
        }
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        return cls(**tables, metadata=metadata)

    @property
    def dimensions(self) -> dict[str, int]:
        return {
            "sources": len(self.sources),
            "destinations": len(self.destinations),
            "products": len(self.products),
            "scenarios": len(self.scenarios),
            "lanes": len(self.lanes),
        }
