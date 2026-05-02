from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class DatasetBundle:
    dataframe: pd.DataFrame
    text_field: str
    label_field: str
    author_field: str

    def split_frame(self, split_name: str) -> pd.DataFrame:
        return self.dataframe[self.dataframe["split"] == split_name].copy()
