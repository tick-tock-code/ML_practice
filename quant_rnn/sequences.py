from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class SampleMetadata:
    ticker: str
    date: str
    target_date: str
    target: float


class EquitySequenceDataset(Dataset):
    """Pooled ticker/date sequence dataset.

    Samples are selected by the row split assigned to the label/target date.
    Sequence context may include earlier rows from another split, but never rows
    after the sample's feature date.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        split: str,
        sequence_length: int,
        feature_columns: list[str] | None = None,
        target_column: str = TARGET_COLUMN,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1.")
        self.split = split
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
        self.target_column = target_column
        self.groups: list[dict[str, Any]] = []
        self.samples: list[tuple[int, int]] = []

        required = {"date", "target_date", "ticker", "split", target_column, *self.feature_columns}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")

        df = frame.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["target_date"] = pd.to_datetime(df["target_date"]).dt.tz_localize(None)
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        for ticker, group in df.groupby("ticker", sort=True):
            group = group.sort_values("date").reset_index(drop=True)
            features = group[self.feature_columns].to_numpy(dtype=np.float32)
            target = group[target_column].to_numpy(dtype=np.float32)
            splits = group["split"].to_numpy()
            dates = group["date"].to_numpy()
            target_dates = group["target_date"].to_numpy()
            group_id = len(self.groups)
            self.groups.append(
                {
                    "ticker": str(ticker),
                    "features": features,
                    "target": target,
                    "splits": splits,
                    "dates": dates,
                    "target_dates": target_dates,
                }
            )
            for row in range(sequence_length - 1, len(group)):
                if splits[row] != split:
                    continue
                x = features[row - sequence_length + 1 : row + 1]
                y = target[row]
                if not np.isfinite(y):
                    continue
                if not np.isfinite(x).all():
                    continue
                self.samples.append((group_id, row))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        group_id, row = self.samples[index]
        group = self.groups[group_id]
        start = row - self.sequence_length + 1
        stop = row + 1
        x = torch.tensor(group["features"][start:stop], dtype=torch.float32)
        y = torch.tensor(group["target"][row], dtype=torch.float32)
        return x, y, torch.tensor(index, dtype=torch.long)

    def sample_metadata(self, index: int) -> SampleMetadata:
        group_id, row = self.samples[index]
        group = self.groups[group_id]
        date = pd.Timestamp(group["dates"][row]).date().isoformat()
        target_date = pd.Timestamp(group["target_dates"][row]).date().isoformat()
        return SampleMetadata(
            ticker=group["ticker"],
            date=date,
            target_date=target_date,
            target=float(group["target"][row]),
        )
