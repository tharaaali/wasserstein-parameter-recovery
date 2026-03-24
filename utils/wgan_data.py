from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Input file not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str], path: str | Path) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{Path(path)} is missing required columns: {missing}")


def load_prepared_training_data(
    fresh_path: str | Path,
    aged_path: str | Path,
    *,
    event_column: str,
    x_column: str,
    y_column: str,
    z_column: str,
    fresh_energy_column: str,
    aged_energy_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_fresh = _read_table(fresh_path)
    df_aged = _read_table(aged_path)

    shared_columns = [event_column, x_column, y_column, z_column]
    _validate_columns(df_fresh, [*shared_columns, fresh_energy_column], fresh_path)
    _validate_columns(df_aged, [*shared_columns, aged_energy_column], aged_path)

    return df_fresh, df_aged


def trim_events_for_debug(
    df: pd.DataFrame,
    *,
    event_column: str,
    max_events: int,
) -> pd.DataFrame:
    keep_events = df[event_column].drop_duplicates().head(max_events)
    return df[df[event_column].isin(keep_events)].copy()


def build_event_tensor_samples(
    df: pd.DataFrame,
    *,
    event_column: str,
    x_column: str,
    y_column: str,
    z_column: str,
    energy_column: str,
    sample_key: str,
    shape: tuple[int, int, int],
) -> list[dict[str, torch.Tensor]]:
    samples: list[dict[str, torch.Tensor]] = []
    grouped = df.groupby(event_column, sort=False)

    for _, rows in grouped:
        calorimeter = torch.zeros(shape, dtype=torch.float32)
        for row in rows.itertuples(index=False):
            x = int(getattr(row, x_column))
            y = int(getattr(row, y_column))
            z = int(getattr(row, z_column))
            energy = float(getattr(row, energy_column))
            calorimeter[z, x, y] = energy
        samples.append({sample_key: calorimeter})

    return samples


def build_target_coords(
    df_fresh: pd.DataFrame,
    df_aged: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    z_column: str,
) -> np.ndarray:
    coords = pd.concat(
        [
            df_fresh[[z_column, x_column, y_column]],
            df_aged[[z_column, x_column, y_column]],
        ],
        ignore_index=True,
    )
    coords = coords.drop_duplicates().astype(int)
    return coords[[z_column, x_column, y_column]].to_numpy()


def load_target_aging_tensor(
    path: str | Path | None,
    *,
    shape: tuple[int, int, int],
    x_column: str,
    y_column: str,
    z_column: str,
    value_column: str,
) -> torch.Tensor | None:
    if path is None:
        return None

    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Target aging file not found: {target_path}")

    suffix = target_path.suffix.lower()
    if suffix == ".pt":
        target = torch.load(target_path, map_location="cpu")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"{target_path} does not contain a torch.Tensor")
        return target.float()
    if suffix == ".npy":
        return torch.from_numpy(np.load(target_path)).float()

    target_df = _read_table(target_path)
    _validate_columns(target_df, [z_column, x_column, y_column, value_column], target_path)

    target_tensor = torch.ones(shape, dtype=torch.float32)
    coord_df = target_df[[z_column, x_column, y_column, value_column]].drop_duplicates()
    for row in coord_df.itertuples(index=False):
        z = int(getattr(row, z_column))
        x = int(getattr(row, x_column))
        y = int(getattr(row, y_column))
        value = float(getattr(row, value_column))
        target_tensor[z, x, y] = value
    return target_tensor
