from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

YAHOO_CHART_TEMPLATE = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?range={data_range}&interval={interval}&events=div%2Csplits"
)


def fetch_yahoo_chart(symbol: str, data_range: str, interval: str, output_path: Path) -> dict:
    """Fetch a Yahoo chart response and save the exact raw bytes.

    Yahoo's chart endpoint is convenient but not a contractual research data API. The
    manifest records the endpoint and hash, and a local JSON snapshot may be supplied
    instead for deterministic reruns.
    """
    url = YAHOO_CHART_TEMPLATE.format(
        symbol=symbol, data_range=data_range, interval=interval
    )
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (research model validation; reproducible local run)"
        },
    )
    raw = urlopen(request, timeout=45).read()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    payload = json.loads(raw.decode("utf-8"))
    return {
        "payload": payload,
        "url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def load_yahoo_chart(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        error = (payload.get("chart") or {}).get("error")
        raise ValueError(f"Yahoo chart payload has no result: {error}")
    record = result[0]
    timestamps = record.get("timestamp") or []
    indicators = record.get("indicators") or {}
    adj_entries = indicators.get("adjclose") or []
    quote_entries = indicators.get("quote") or []
    if not adj_entries:
        raise ValueError("Adjusted-close data are absent; raw close is not accepted")
    adjusted_close = adj_entries[0].get("adjclose") or []
    volume = (quote_entries[0].get("volume") or []) if quote_entries else []
    if len(timestamps) != len(adjusted_close):
        raise ValueError("Timestamp and adjusted-close arrays are misaligned")
    rows: list[dict] = []
    for idx, (timestamp, price) in enumerate(zip(timestamps, adjusted_close)):
        if price is None or not np.isfinite(float(price)):
            continue
        row = {
            "date": pd.to_datetime(int(timestamp), unit="s", utc=True).normalize(),
            "adjusted_close": float(price),
        }
        if idx < len(volume) and volume[idx] is not None:
            row["volume"] = float(volume[idx])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) < 1000:
        raise ValueError(f"Insufficient valid observations: {len(frame)}")
    return (
        frame.drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_adjusted_close_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "adjusted_close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain {sorted(required)}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    frame = frame.dropna(subset=["date", "adjusted_close"])
    return frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
