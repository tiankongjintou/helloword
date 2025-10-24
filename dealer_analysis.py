"""Utilities for evaluating the strength and style of Chinese A-share stock operators.

The module loads historical trading data for individual stocks and derives a set of
quantitative metrics for every major bull market since 2013 as well as resilience
metrics during crash periods.  It then combines those metrics into a composite score
and textual rating (S/A/B/C/D) that approximates the relative strength of the major
shareholder or "dealer" behind each stock.

Example
-------
>>> python dealer_analysis.py --data data_dir/ --output dealer_scores.xlsx

The input may be a single CSV or a directory/glob of per-stock CSV files. Each table
is expected to contain the Tushare-style columns listed in the prompt, including at
minimum: ``ts_code`` (ticker), ``trade_date`` (YYYYMMDD), and ``close``. Additional
turnover and volume columns will be used when present.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketPeriod:
    """Represents a named market window.

    Attributes
    ----------
    name:
        Short name for the period, used as a column prefix in the output.
    start:
        Inclusive start date of the window (``YYYY-MM-DD``).
    end:
        Inclusive end date of the window (``YYYY-MM-DD``).
    narrative:
        Human readable logic behind the period.  This is returned in the metadata so
        downstream analysts can understand why the period exists.
    post_window:
        Number of trading days after ``end`` to inspect for drawdown (bull markets
        only).  ``None`` disables the post analysis.
    """

    name: str
    start: str
    end: str
    narrative: str
    post_window: Optional[int] = None


# Major bull markets for the CSI 300 / Shanghai Composite since 2007.
BULL_MARKETS: List[MarketPeriod] = [
    MarketPeriod(
        name="2007超级牛市",
        start="2006-08-01",
        end="2007-10-16",
        narrative="股改完成与经济高速增长驱动的2007年超级牛市",
        post_window=60,
    ),
    MarketPeriod(
        name="2014-2015杠杆牛",
        start="2014-07-01",
        end="2015-06-12",
        narrative="全面改革、融资融券和伞形信托推动的加杠杆牛市",
        post_window=60,
    ),
    MarketPeriod(
        name="2019科创反弹",
        start="2019-01-04",
        end="2019-04-19",
        narrative="科创板预期和货币宽松触发的春季行情",
        post_window=40,
    ),
    MarketPeriod(
        name="2020-2021疫后修复",
        start="2020-03-24",
        end="2021-02-18",
        narrative="全球流动性宽松与新经济龙头共振的疫后修复牛市",
        post_window=60,
    ),
    MarketPeriod(
        name="2022-2023信心修复",
        start="2022-10-31",
        end="2023-05-08",
        narrative="疫情防控优化与地产、数字经济政策驱动的修复行情",
        post_window=40,
    ),
]

# Major crash periods to gauge resilience.
CRASH_PERIODS: List[MarketPeriod] = [
    MarketPeriod(
        name="2015股灾",
        start="2015-06-15",
        end="2016-01-27",
        narrative="杠杆去化引发的市场快速去泡沫",
    ),
    MarketPeriod(
        name="2018去杠杆",
        start="2018-01-29",
        end="2018-10-19",
        narrative="去杠杆与贸易摩擦导致的系统性下跌",
    ),
    MarketPeriod(
        name="2022疫情冲击",
        start="2022-01-04",
        end="2022-10-31",
        narrative="疫情反复与全球紧缩的冲击窗口",
    ),
]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df["trade_date"].dtype, np.datetime64):
        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
    return df


def _first_value(series: pd.Series) -> float:
    return float(series.iloc[0]) if not series.empty else math.nan


def _last_value(series: pd.Series) -> float:
    return float(series.iloc[-1]) if not series.empty else math.nan


def _normalize(value: float, lower: float, upper: float) -> float:
    if math.isnan(value):
        return 0.5
    if lower == upper:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _analyze_bull_market(stock_df: pd.DataFrame, period: MarketPeriod) -> Dict[str, float]:
    window = stock_df.loc[
        (stock_df["trade_date"] >= period.start) & (stock_df["trade_date"] <= period.end)
    ].copy()
    if window.empty:
        return {
            "start_close": math.nan,
            "peak_close": math.nan,
            "peak_date": pd.NaT,
            "return_multiple": math.nan,
            "avg_turnover_rate": math.nan,
            "avg_volume_ratio": math.nan,
            "participation_score": math.nan,
            "post_drawdown": math.nan,
        }

    window.sort_values("trade_date", inplace=True)
    start_close = _first_value(window["close"])
    peak_idx = window["close"].idxmax()
    peak_close = float(window.loc[peak_idx, "close"])
    peak_date = window.loc[peak_idx, "trade_date"]
    return_multiple = peak_close / start_close if start_close else math.nan

    turnover_mean = window.get("turnover_rate", pd.Series(dtype=float)).mean()
    volume_ratio_mean = window.get("volume_ratio", pd.Series(dtype=float)).mean()
    # Participation score balances trading enthusiasm and liquidity rotation.
    participation_score = np.nan
    if not math.isnan(turnover_mean) or not math.isnan(volume_ratio_mean):
        turnover_norm = _normalize(turnover_mean, 0.0, 8.0)
        volume_norm = _normalize(volume_ratio_mean, 0.5, 2.0)
        participation_score = 0.6 * turnover_norm + 0.4 * volume_norm

    post_drawdown = math.nan
    if period.post_window:
        post_window = stock_df.loc[
            (stock_df["trade_date"] > period.end)
            & (stock_df["trade_date"] <= pd.to_datetime(period.end) + pd.tseries.offsets.BDay(period.post_window))
        ].copy()
        if not post_window.empty:
            post_window.sort_values("trade_date", inplace=True)
            trough_close = float(post_window["close"].min())
            post_drawdown = trough_close / peak_close - 1.0

    return {
        "start_close": start_close,
        "peak_close": peak_close,
        "peak_date": peak_date,
        "return_multiple": return_multiple,
        "avg_turnover_rate": turnover_mean,
        "avg_volume_ratio": volume_ratio_mean,
        "participation_score": participation_score,
        "post_drawdown": post_drawdown,
    }


def _analyze_crash(stock_df: pd.DataFrame, period: MarketPeriod) -> Dict[str, float]:
    window = stock_df.loc[
        (stock_df["trade_date"] >= period.start) & (stock_df["trade_date"] <= period.end)
    ].copy()
    if window.empty:
        return {
            "start_close": math.nan,
            "end_close": math.nan,
            "max_drawdown": math.nan,
            "recovery_ratio": math.nan,
            "resilience_score": math.nan,
        }

    window.sort_values("trade_date", inplace=True)
    start_close = _first_value(window["close"])
    end_close = _last_value(window["close"])
    trough_close = float(window["close"].min())
    max_drawdown = trough_close / start_close - 1.0
    recovery_ratio = end_close / trough_close - 1.0 if trough_close else math.nan

    depth_score = np.clip(1.0 + max_drawdown, 0.0, 1.0)
    recovery_score = np.clip(recovery_ratio / 0.5, 0.0, 1.0)
    resilience_score = float(0.7 * depth_score + 0.3 * recovery_score)

    return {
        "start_close": start_close,
        "end_close": end_close,
        "max_drawdown": max_drawdown,
        "recovery_ratio": recovery_ratio,
        "resilience_score": resilience_score,
    }


def _summarize_scores(stock_metrics: Dict[str, float]) -> Dict[str, float]:
    bull_multiples = [
        value
        for key, value in stock_metrics.items()
        if key.endswith("return_multiple") and not math.isnan(value)
    ]
    bull_participation = [
        value
        for key, value in stock_metrics.items()
        if key.endswith("participation_score") and not math.isnan(value)
    ]
    bull_drawdowns = [
        value
        for key, value in stock_metrics.items()
        if key.endswith("post_drawdown") and not math.isnan(value)
    ]
    crash_scores = [
        value
        for key, value in stock_metrics.items()
        if key.endswith("resilience_score") and not math.isnan(value)
    ]

    avg_multiple = float(np.nanmean(bull_multiples)) if bull_multiples else math.nan
    avg_participation = float(np.nanmean(bull_participation)) if bull_participation else math.nan
    avg_drawdown = float(np.nanmean(bull_drawdowns)) if bull_drawdowns else math.nan
    avg_resilience = float(np.nanmean(crash_scores)) if crash_scores else math.nan

    multiple_score = _normalize(avg_multiple, 1.0, 5.0)
    participation_score = avg_participation if not math.isnan(avg_participation) else 0.5
    drawdown_score = _normalize(-avg_drawdown if not math.isnan(avg_drawdown) else math.nan, 0.0, 0.5)
    stress_score = avg_resilience if not math.isnan(avg_resilience) else 0.5

    composite = (
        0.4 * multiple_score
        + 0.3 * participation_score
        + 0.2 * drawdown_score
        + 0.1 * stress_score
    )

    if composite >= 0.85:
        rating = "S"
    elif composite >= 0.7:
        rating = "A"
    elif composite >= 0.55:
        rating = "B"
    elif composite >= 0.4:
        rating = "C"
    else:
        rating = "D"

    return {
        "avg_bull_multiple": avg_multiple,
        "avg_participation_score": avg_participation,
        "avg_post_drawdown": avg_drawdown,
        "avg_crash_resilience": avg_resilience,
        "composite_score": composite,
        "rating": rating,
    }


def evaluate_dealer_strength(
    data: pd.DataFrame,
    bull_markets: Iterable[MarketPeriod] = BULL_MARKETS,
    crash_periods: Iterable[MarketPeriod] = CRASH_PERIODS,
) -> pd.DataFrame:
    """Compute dealer strength metrics for every stock in ``data``.

    Parameters
    ----------
    data:
        Historical trading dataset containing one row per stock per trading day.
    bull_markets:
        Iterable of bull market windows to evaluate.
    crash_periods:
        Iterable of crash windows used to score stress resilience.

    Returns
    -------
    pandas.DataFrame
        A wide table that lists per-stock metrics for each period along with the
        composite score and rating.
    """

    df = _ensure_datetime(data)
    df.sort_values(["ts_code", "trade_date"], inplace=True)

    results: List[Dict[str, object]] = []
    for ts_code, stock_df in df.groupby("ts_code", sort=False):
        stock_metrics: Dict[str, object] = {"ts_code": ts_code}
        for period in bull_markets:
            metrics = _analyze_bull_market(stock_df, period)
            prefix = f"{period.name}"
            for key, value in metrics.items():
                stock_metrics[f"{prefix}_{key}"] = value
            stock_metrics[f"{prefix}_narrative"] = period.narrative

        for period in crash_periods:
            metrics = _analyze_crash(stock_df, period)
            prefix = f"{period.name}"
            for key, value in metrics.items():
                stock_metrics[f"{prefix}_{key}"] = value
            stock_metrics[f"{prefix}_narrative"] = period.narrative

        stock_metrics.update(_summarize_scores(stock_metrics))
        results.append(stock_metrics)

    return pd.DataFrame(results)


def _load_trade_data(path: str) -> pd.DataFrame:
    """Load one or many per-stock trading files into a single dataframe."""

    resource = Path(path)
    if resource.is_dir():
        frames: List[pd.DataFrame] = []
        for csv_path in sorted(resource.rglob("*.csv")):
            frame = pd.read_csv(csv_path)
            if "ts_code" not in frame.columns:
                raise ValueError(
                    f"Missing 'ts_code' column in {csv_path}. Each file must identify the stock."
                )
            frames.append(frame)
        if not frames:
            raise FileNotFoundError(f"No CSV files found under directory: {resource}")
        return pd.concat(frames, ignore_index=True, sort=False)

    if resource.is_file():
        frame = pd.read_csv(resource)
        if "ts_code" not in frame.columns:
            raise ValueError(
                f"Missing 'ts_code' column in {resource}. Provide Tushare-style data with identifiers."
            )
        return frame

    matches = list(Path().glob(path))
    if matches:
        frames = []
        for match in matches:
            if match.is_file():
                frame = pd.read_csv(match)
                if "ts_code" not in frame.columns:
                    raise ValueError(
                        f"Missing 'ts_code' column in {match}. Provide Tushare-style data with identifiers."
                    )
                frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True, sort=False)
    raise FileNotFoundError(f"Unable to locate input data using path: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A-share dealer strength.")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to a CSV file, directory of per-stock CSVs, or glob pattern.",
    )
    parser.add_argument(
        "--output",
        default="dealer_strength.xlsx",
        help="Path to save the resulting report. Defaults to 'dealer_strength.xlsx'.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top stocks to display in the console preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = _load_trade_data(args.data)
    report = evaluate_dealer_strength(data)

    output_path = Path(args.output)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        report.to_excel(output_path, index=False)
    else:
        report.to_csv(output_path, index=False)
    print(f"Saved dealer strength report to {output_path}")

    preview = report.sort_values("composite_score", ascending=False).head(args.top)
    pd.set_option("display.max_columns", None)
    print(preview)


if __name__ == "__main__":
    main()
