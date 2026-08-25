from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-01", periods=1200, freq="B", tz="UTC")
    returns = rng.normal(0.0002, 0.01, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"date": dates, "adjusted_close": prices})
