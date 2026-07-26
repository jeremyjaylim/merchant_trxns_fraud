import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class DebitTransactionSchema(pa.DataFrameModel):
    # Primary identifiers
    transaction_id: Series[str] = pa.Field(
        unique=True,
        nullable=False,
    )

    # Timestamps (Bounded between Jan 1, 2020 and present time)
    post_ts: Series[pa.DateTime] = pa.Field(
        nullable=False,
        ge=pd.Timestamp("2020-01-01"),
        le=pd.Timestamp.now(),
    )

    # Identifiers with pattern validation
    customer_id: Series[str] = pa.Field(
        str_matches=r"^C\d+$",
        nullable=False,
    )

    # Terminal ID is optional (e.g., null for Online entry mode)
    terminal_id: Series[str] = pa.Field(
        str_matches=r"^T\d+$",
        nullable=True,
    )

    # Financial amount (allows $0 auth holds while capping extreme errors)
    amt: Series[float] = pa.Field(
        ge=0,
        le=100000.0,
        nullable=False,
    )

    # Allowed transaction types
    entry_mode: Series[str] = pa.Field(
        isin=["Swipe", "Chip", "Contactless", "Online"],
        nullable=False,
    )

    # Aggregated daily feature values (must be non-negative)
    Amt_Avg_Daily: Series[float] = pa.Field(ge=0, nullable=False)
    Amount_StdDevn_Daily: Series[float] = pa.Field(ge=0, nullable=False)
    TrxnCount_Avg_Daily: Series[float] = pa.Field(ge=0, nullable=False)

    # Merchant details (ge=1 allows 3-digit MCC codes like 742 or 780)
    mcc: Series[int] = pa.Field(ge=1, le=9999, nullable=False)
    MerchantType: Series[str] = pa.Field(nullable=False)

    # Custom Cross-Column Rule: Terminal ID should be null/empty if Entry Mode is Online
    @pa.dataframe_check
    def check_online_terminal(cls, df: pd.DataFrame) -> Series[bool]:
        """Ensure Online transactions do not have physical Terminal IDs assigned."""
        online_mask = df["entry_mode"] == "Online"
        return ~online_mask | df["terminal_id"].isna()

    class Config:
        coerce = True
        strict = True

