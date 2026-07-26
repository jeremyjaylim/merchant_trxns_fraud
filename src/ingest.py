from pathlib import Path
import pandas as pd
from src.data_validation import DebitTransactionSchema

# Define the default data directory path relative to the project root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw_data(file_name: str = "debit_transactions.csv") -> pd.DataFrame:
    """Reads raw transaction data from the local \\data\\ folder."""
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    print(f"Reading source data from: {file_path}")

    # Read based on file extension
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def ingest_and_validate(file_name: str = "debit_transactions.csv") -> pd.DataFrame:
    """Loads data from \\data\\ and validates it against DebitTransactionSchema."""
    # 1. Load data
    raw_df = load_raw_data(file_name)

    # 2. Validate data schema
    print("Validating data against schema...")
    validated_df = DebitTransactionSchema.validate(raw_df)

    print(
        f"Successfully ingested and validated {len(validated_df)} rows of data."
    )
    return validated_df


if __name__ == "__main__":
    # Example usage for testing locally:
    # Replace 'debit_transactions.csv' with your actual file name inside \data\
    df = ingest_and_validate("debit_transactions.csv")
    print(df.head())