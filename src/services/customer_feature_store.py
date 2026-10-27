"""
============================================================
CreditLens-AI
Customer Feature Store
============================================================

Server-side lookup of a customer's production feature record.

Analysts identify a customer by ID; the backend retrieves the
production features itself, so no feature data ever has to be
typed or pasted into the UI.

Current source: ``data/processed/test_processed.parquet``,
indexed by ``SK_ID_CURR``. This is a development/demo data
source. A production deployment would replace the loading
logic in this class with a database or feature-store read;
the public interface would stay the same.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.paths import PROCESSED_DATA_DIR


DEFAULT_DATA_PATH = (
    PROCESSED_DATA_DIR / "test_processed.parquet"
)

ID_COLUMN = "SK_ID_CURR"


class CustomerNotFoundError(LookupError):
    """Raised when a customer ID is not in the feature store."""


class FeatureStoreUnavailableError(RuntimeError):
    """Raised when the feature data source cannot be read."""


class CustomerFeatureStore:
    """
    Lazy, read-only customer feature lookup.
    """

    def __init__(
        self,
        data_path: Path | None = None,
        id_column: str = ID_COLUMN,
    ) -> None:

        self.data_path = (
            data_path if data_path is not None
            else DEFAULT_DATA_PATH
        )

        self.id_column = id_column

        self._frame: pd.DataFrame | None = None

        self._lock = threading.Lock()

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def _load(self) -> pd.DataFrame:

        if self._frame is not None:
            return self._frame

        with self._lock:

            if self._frame is not None:
                return self._frame

            if not self.data_path.exists():
                raise FeatureStoreUnavailableError(
                    "Customer feature data source not found."
                )

            frame = pd.read_parquet(self.data_path)

            if self.id_column not in frame.columns:
                raise FeatureStoreUnavailableError(
                    "Customer identifier column is missing "
                    "from the feature data source."
                )

            ids = frame[self.id_column]

            if pd.api.types.is_float_dtype(ids):
                ids = ids.astype("int64")

            frame.index = ids.astype(str)

            frame = frame[~frame.index.duplicated(keep="first")]

            self._frame = frame

            return self._frame

    # --------------------------------------------------------
    # Lookup
    # --------------------------------------------------------

    def get_features(
        self,
        customer_id: str,
    ) -> dict[str, float | int | str | None]:
        """
        Return the customer's production features.

        Null values are returned as ``None``. The identifier
        column is not part of the feature record.
        """

        frame = self._load()

        key = str(customer_id).strip()

        if key not in frame.index:
            raise CustomerNotFoundError(key)

        record = (
            frame.loc[[key]]
            .drop(columns=[self.id_column])
            .to_dict(orient="records")[0]
        )

        return {
            column: self._clean(value)
            for column, value in record.items()
        }

    def sample_customer_ids(
        self,
        limit: int = 20,
    ) -> list[str]:
        """Return up to ``limit`` available customer IDs."""

        frame = self._load()

        return [str(i) for i in frame.index[:limit]]

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _clean(value: Any) -> Any:

        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if hasattr(value, "item"):
            return value.item()

        return value
