from pathlib import Path
from faf_fctp.catalogue import generate_candidate_catalogue, load_design

if __name__ == "__main__":
    config = load_design()
    output = Path("data/catalogue/candidate_catalogue.parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_candidate_catalogue(config).to_parquet(output, index=False, compression="zstd", engine="pyarrow")
    print(output)
