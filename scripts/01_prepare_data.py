"""Step 1: UWHVF -> study cohort, samples and folds.

    uv run python scripts/01_prepare_data.py

Reads data/raw/ and writes:
    outputs/exams.parquet     filtered exams, with block_id and fold
    outputs/samples.parquet   one row per (history -> next exam)
    outputs/grid.parquet      geometry of the 54 locations
    outputs/filter_log.csv    what each filter F1..F5 removed
"""

import hashlib

import polars as pl

from src import config, data

# sha256 of the UWHVF files used in the paper. A mismatch means a different dataset version.
EXPECTED_SHA256 = {
    "alldata.json": "0ac2e6059daa9158912092b8ebd0852be719950b3e87b8be827524bb12271d74",
    "Coord_242.csv": "734fbd34485fff758420b7b7202fe8a147fed1f79594b632dd49c20bd99732cc",
    "Series_Summary.csv": "e43475afd62ac8a06f6204746574425aff95106f2c5c56db7a417d6fcb23509f",
}


def check_raw_files() -> None:
    for name, expected in EXPECTED_SHA256.items():
        path = config.RAW_DIR / name
        if not path.exists():
            raise SystemExit(f"missing {path}. See data/README.md.")
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != expected:
            raise SystemExit(f"{name}: sha256 {found} differs from the paper's {expected}")


def main() -> None:
    check_raw_files()
    config.OUT_DIR.mkdir(exist_ok=True)

    exams, log = data.apply_filters(data.load_exams())
    folds = data.assign_folds(exams)
    exams = exams.join(folds, on="pat_id", how="left")
    samples = data.build_samples(exams).join(folds, on="pat_id", how="left")

    exams.write_parquet(config.EXAMS_PARQUET)
    samples.write_parquet(config.SAMPLES_PARQUET)
    data.load_grid().write_parquet(config.GRID_PARQUET)
    log.write_csv(config.FILTER_LOG_CSV)

    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",  # the Windows console is not UTF-8
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
        tbl_width_chars=120,
    ):
        print(log)
        print(f"\nsamples: {samples.height}  (evaluation, last of each series: {samples['is_last'].sum()})")
        print(folds.group_by("fold").len().sort("fold").rename({"len": "patients"}))


if __name__ == "__main__":
    main()
