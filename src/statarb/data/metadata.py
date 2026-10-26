import json
from datetime import datetime
from pathlib import Path


def create_metadata(
    tickers: list[str],
    start_date: str,
    end_date: str,
    price_adjustment: str = "auto_adjust=True",
) -> dict:
    """Create metadata describing a downloaded price dataset."""
    if not tickers or all(not ticker.strip() for ticker in tickers):
        raise ValueError("At least one valid ticker is required.")

    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date.")

    return {
        "source": "yfinance",
        "tickers": [ticker.strip().upper() for ticker in tickers],
        "start_date": start_date,
        "end_date": end_date,
        "downloaded_at": datetime.now().astimezone().isoformat(),
        "price_adjustment": price_adjustment,
    }


def save_metadata(metadata: dict, path: str | Path) -> None:
    """Save dataset metadata as JSON."""
    required_fields = {
        "source",
        "tickers",
        "start_date",
        "end_date",
        "downloaded_at",
        "price_adjustment",
    }

    if not required_fields.issubset(metadata):
        raise ValueError("Metadata is missing required fields.") # interesting how we can compare set with dictionaries

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_metadata(path: str | Path) -> dict:
    """Load and validate basic metadata structure from JSON."""
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"Metadata file not found: {source}")

    try:
        with source.open(encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid metadata JSON.") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object.")

    required_fields = {
        "source",
        "tickers",
        "start_date",
        "end_date",
        "downloaded_at",
        "price_adjustment",
    }

    if not required_fields.issubset(metadata):
        raise ValueError("Metadata is missing required fields.")

    return metadata
