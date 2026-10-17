"""FAF forecast ingestion with an explicit empirical/synthetic boundary."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dms_orig",
    "dms_dest",
    "dms_mode",
    "sctg2",
    "trade_type",
    "tons_2030",
    "tons_2030_low",
    "tons_2030_high",
    "tmiles_2030",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bundled_names(source: Path) -> tuple[dict[str, str], dict[str, str]]:
    zone_names: dict[str, str] = {}
    product_names: dict[str, str] = {}
    packaged_codebook_directory = Path(__file__).resolve().parent / "resources"
    zone_path = source.parent / "codebook_faf_zones_domestic.parquet"
    product_path = source.parent / "codebook_sctg2_commodities.parquet"
    if not zone_path.exists():
        zone_path = packaged_codebook_directory / "codebook_faf_zones_domestic.parquet"
    if not product_path.exists():
        product_path = packaged_codebook_directory / "codebook_sctg2_commodities.parquet"
    try:
        if zone_path.exists():
            zones = pd.read_parquet(zone_path, engine="pyarrow").astype("string")
            if {"Numeric Label", "Short Description"}.issubset(zones.columns):
                zone_names = {
                    "O" + str(code).zfill(3): str(name)
                    for code, name in zip(zones["Numeric Label"], zones["Short Description"])
                }
                zone_names.update({"D" + key[1:]: value for key, value in list(zone_names.items())})
        if product_path.exists():
            products = pd.read_parquet(product_path, engine="pyarrow").astype("string")
            if {"Numeric Label", "Description"}.issubset(products.columns):
                product_names = {
                    "SCTG" + str(code).zfill(2): str(name)
                    for code, name in zip(products["Numeric Label"], products["Description"])
                }
    except (ImportError, ModuleNotFoundError):
        pass
    return zone_names, product_names


def _read_source(
    path: Path,
    usecols: set[str],
    *,
    domestic_mode: str,
    trade_type: str,
) -> pd.DataFrame:
    dtype = {
        "dms_orig": "string",
        "dms_dest": "string",
        "dms_mode": "string",
        "sctg2": "string",
        "trade_type": "string",
    }
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path, columns=sorted(usecols), engine="pyarrow")
        for column, column_type in dtype.items():
            frame[column] = frame[column].astype(column_type)
        return frame
    if path.suffix.lower() != ".zip":
        raise ValueError(f"unsupported FAF source {path}; use the official .zip or a .parquet fixture")
    with zipfile.ZipFile(path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv") and "readme" not in name.lower()
        ]
        preferred = [name for name in members if "hilo" in name.lower() or "forecast" in name.lower()]
        choices = preferred or members
        if not choices:
            raise ValueError(f"no CSV file found inside {path}")
        selected_member = sorted(
            choices, key=lambda name: archive.getinfo(name).file_size, reverse=True
        )[0]
        with archive.open(selected_member) as handle:
            selected_chunks: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                handle,
                dtype=dtype,
                usecols=usecols,
                low_memory=False,
                chunksize=200_000,
            ):
                for column in ("dms_mode", "trade_type"):
                    chunk[column] = (
                        chunk[column]
                        .astype("string")
                        .str.strip()
                        .str.replace(r"\.0$", "", regex=True)
                    )
                selected_chunks.append(
                    chunk.loc[
                        (chunk["dms_mode"] == str(domestic_mode))
                        & (chunk["trade_type"] == str(trade_type))
                    ].copy()
                )
            if not selected_chunks:
                return pd.DataFrame(columns=sorted(usecols))
            return pd.concat(selected_chunks, ignore_index=True)



def verify_faf_source(path: str | Path, config: dict[str, object]) -> dict[str, str]:
    """Verify the exact FAF archive identity required by the benchmark design."""
    source = Path(path)
    expected = config["faf_source"]
    expected_name = str(expected["archive_name"])
    expected_sha256 = str(expected["sha256"]).lower()
    if source.name != expected_name:
        raise ValueError(f"expected FAF archive {expected_name!r}; received {source.name!r}")
    observed = _sha256_file(source).lower()
    if observed != expected_sha256:
        raise ValueError(
            "FAF source SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed}"
        )
    return {"archive_name": source.name, "sha256": observed}

def load_faf_forecast_flows(
    path: str | Path,
    *,
    year: int = 2030,
    domestic_mode: str = "1",
    trade_type: str = "1",
) -> pd.DataFrame:
    """Load domestic FAF flows into the normalised empirical flow schema."""

    source = Path(path)
    year_columns = {
        "base": f"tons_{year}",
        "low": f"tons_{year}_low",
        "high": f"tons_{year}_high",
        "tmiles": f"tmiles_{year}",
    }
    required = (REQUIRED_COLUMNS - {"tons_2030", "tons_2030_low", "tons_2030_high", "tmiles_2030"}) | set(
        year_columns.values()
    )
    raw = _read_source(
        source,
        required,
        domestic_mode=str(domestic_mode),
        trade_type=str(trade_type),
    )
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"FAF source is missing required columns: {missing}")

    for column in ("dms_orig", "dms_dest", "dms_mode", "sctg2", "trade_type"):
        raw[column] = raw[column].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    raw["dms_orig"] = raw["dms_orig"].str.zfill(3)
    raw["dms_dest"] = raw["dms_dest"].str.zfill(3)
    raw["sctg2"] = raw["sctg2"].str.zfill(2)
    selected = raw.loc[
        (raw["dms_mode"] == str(domestic_mode))
        & (raw["trade_type"] == str(trade_type))
        & raw["dms_orig"].notna()
        & raw["dms_dest"].notna()
    ].copy()
    if selected.empty:
        raise ValueError(f"no FAF records match dms_mode={domestic_mode!r}, trade_type={trade_type!r}")

    normalized = pd.DataFrame(
        {
            "source_id": "O" + selected["dms_orig"],
            "destination_id": "D" + selected["dms_dest"],
            "product_id": "SCTG" + selected["sctg2"],
            "tons_low_ktons": pd.to_numeric(selected[year_columns["low"]], errors="coerce").fillna(0.0),
            "tons_base_ktons": pd.to_numeric(selected[year_columns["base"]], errors="coerce").fillna(0.0),
            "tons_high_ktons": pd.to_numeric(selected[year_columns["high"]], errors="coerce").fillna(0.0),
            "ton_miles_base_millions": pd.to_numeric(selected[year_columns["tmiles"]], errors="coerce").fillna(0.0),
        }
    )
    numeric = [column for column in normalized if column.endswith("ktons") or column.endswith("millions")]
    normalized[numeric] = normalized[numeric].clip(lower=0.0)
    normalized = (
        normalized.groupby(["source_id", "destination_id", "product_id"], as_index=False, sort=True)[numeric]
        .sum()
        .sort_values(["product_id", "source_id", "destination_id"])
        .reset_index(drop=True)
    )
    normalized["distance_miles"] = np.divide(
        normalized["ton_miles_base_millions"] * 1000.0,
        normalized["tons_base_ktons"],
        out=np.zeros(len(normalized), dtype=float),
        where=normalized["tons_base_ktons"].to_numpy() > 0,
    )
    normalized["distance_miles"] = normalized["distance_miles"].clip(lower=1.0)
    zone_names, product_names = _load_bundled_names(source)
    normalized.attrs.update(
        {
            "source_filename": source.name,
            "source_sha256": _sha256_file(source),
            "forecast_year": int(year),
            "domestic_mode": str(domestic_mode),
            "trade_type": str(trade_type),
            "zone_names": zone_names,
            "product_names": product_names,
        }
    )
    return normalized


def inspect_faf_source(path: str | Path, *, year: int = 2030) -> dict[str, object]:
    flows = load_faf_forecast_flows(path, year=year)
    return {
        "records": int(len(flows)),
        "sources": int(flows["source_id"].nunique()),
        "destinations": int(flows["destination_id"].nunique()),
        "products": int(flows["product_id"].nunique()),
        "product_ids": sorted(flows["product_id"].unique().tolist()),
        "base_volume_ktons": float(flows["tons_base_ktons"].sum()),
    }
