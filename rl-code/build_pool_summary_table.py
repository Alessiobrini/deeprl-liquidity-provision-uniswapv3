from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import dotenv_values


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "paper_figures"

SUBGRAPH_ID = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

POOL_CONFIGS = {
    "WETH/USDC": {
        "address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        "fee_tier": "0.05\\%",
        "price_csv": BASE_DIR / "data_price_uni_h_time.csv",
        "config_yaml": BASE_DIR / "config" / "uniswap_rl_param_1108.yaml",
    },
    "WBTC/USDC": {
        "address": "0x99ac8cA7087fa4A2A1FB6357269965A2014ABc35",
        "fee_tier": "0.30\\%",
        "price_csv": BASE_DIR / "data" / "data_price_wbtc_usdc_030_h_time.csv",
        "config_yaml": BASE_DIR / "config" / "pool_wbtc_usdc_030.yaml",
    },
}


def graph_api_url() -> str:
    env_path = BASE_DIR / "data" / ".env"
    cfg = dotenv_values(env_path)
    api_key = cfg.get("THEGRAPH_API_KEY") or os.getenv("THEGRAPH_API_KEY")
    if not api_key:
        raise RuntimeError("THEGRAPH_API_KEY not found in rl-code/data/.env or environment.")
    return f"https://gateway.thegraph.com/api/{api_key}/subgraphs/id/{SUBGRAPH_ID}"


def fetch_pool_hour_metrics(pool_address: str) -> pd.DataFrame:
    url = graph_api_url()
    all_rows: list[dict] = []
    last_timestamp = 0
    batch = 0

    while True:
        query = f"""
        {{
          poolHourDatas(
            first: 1000,
            where: {{pool: "{pool_address.lower()}", periodStartUnix_gt: {last_timestamp}}},
            orderBy: periodStartUnix,
            orderDirection: asc
          ) {{
            periodStartUnix
            volumeUSD
            feesUSD
            txCount
          }}
        }}
        """

        response = requests.post(url, json={"query": query}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"Graph query failed: {payload['errors']}")

        rows = payload.get("data", {}).get("poolHourDatas", [])
        if not rows:
            break

        all_rows.extend(rows)
        last_timestamp = int(rows[-1]["periodStartUnix"])
        batch += 1
        print(
            f"Fetched pool {pool_address[:8]}... batch {batch}, total rows {len(all_rows):,}, "
            f"last ts {pd.to_datetime(last_timestamp, unit='s')}",
            flush=True,
        )

    if not all_rows:
        raise RuntimeError(f"No hourly rows returned for pool {pool_address}")

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["periodStartUnix"].astype(int), unit="s")
    for col in ["volumeUSD", "feesUSD", "txCount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["timestamp", "volumeUSD", "feesUSD", "txCount"]].sort_values("timestamp")


def load_used_price_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def align_metrics_to_used_hours(metrics_df: pd.DataFrame, used_price_df: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics_df.set_index("timestamp").sort_index()
    used_idx = pd.DatetimeIndex(used_price_df["timestamp"])

    full_idx = pd.date_range(start=used_idx.min(), end=used_idx.max(), freq="h")
    metrics = metrics.reindex(full_idx)
    metrics[["volumeUSD", "feesUSD", "txCount"]] = metrics[["volumeUSD", "feesUSD", "txCount"]].fillna(0.0)
    metrics = metrics.loc[used_idx]
    metrics.index.name = "timestamp"
    return metrics.reset_index()


def compute_realized_vol(price_df: pd.DataFrame) -> tuple[float, float]:
    log_returns = np.log(price_df["price"]).diff().dropna()
    hourly_vol = float(log_returns.std())
    annualized_vol = float(hourly_vol * np.sqrt(24 * 365))
    return hourly_vol, annualized_vol


def load_gas_assumption(config_path: Path) -> float:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return float(cfg["gas_fee"])


def build_summary() -> pd.DataFrame:
    rows = []
    for pool_name, cfg in POOL_CONFIGS.items():
        print(f"\nBuilding summary for {pool_name}...", flush=True)
        price_df = load_used_price_series(cfg["price_csv"])
        metrics_raw = fetch_pool_hour_metrics(cfg["address"])
        metrics_used = align_metrics_to_used_hours(metrics_raw, price_df)
        hourly_vol, annualized_vol = compute_realized_vol(price_df)
        gas_fee = load_gas_assumption(cfg["config_yaml"])
        mean_volume = float(metrics_used["volumeUSD"].mean())
        mean_fees = float(metrics_used["feesUSD"].mean())
        mean_swaps = float(metrics_used["txCount"].mean())

        # The current subgraph endpoint returns zeroed USD volume/fee fields for
        # the WETH/USDC hourly buckets even when txCount is positive. Treat those
        # fields as unavailable rather than as true zeros.
        if mean_volume == 0.0 and mean_fees == 0.0 and mean_swaps > 0:
            mean_volume = np.nan
            mean_fees = np.nan

        rows.append(
            {
                "Pool": pool_name,
                "Fee tier": cfg["fee_tier"],
                "Start date": price_df["timestamp"].min().strftime("%-d %b %Y"),
                "End date": price_df["timestamp"].max().strftime("%-d %b %Y"),
                "Hours": int(len(price_df)),
                "Mean hourly volume (USD)": mean_volume,
                "Mean hourly swaps": mean_swaps,
                "Hourly realized vol (%)": hourly_vol * 100,
                "Annualized realized vol (%)": annualized_vol * 100,
                "Mean hourly fee revenue (USD)": mean_fees,
                "Gas / rebalance assumption": gas_fee,
            }
        )

    return pd.DataFrame(rows)


def write_outputs(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "pool_summary_table.csv"
    summary.to_csv(csv_path, index=False)

    latex_df = summary.copy()
    def fmt_or_na(x: float, decimals: int = 2) -> str:
        if pd.isna(x):
            return "N/A"
        return f"{x:,.{decimals}f}"

    for col in [
        "Mean hourly volume (USD)",
        "Mean hourly swaps",
        "Hourly realized vol (%)",
        "Annualized realized vol (%)",
        "Mean hourly fee revenue (USD)",
        "Gas / rebalance assumption",
    ]:
        if col == "Mean hourly swaps":
            latex_df[col] = latex_df[col].map(lambda x: fmt_or_na(x, 2))
        elif col == "Gas / rebalance assumption":
            latex_df[col] = latex_df[col].map(lambda x: fmt_or_na(x, 0))
        else:
            latex_df[col] = latex_df[col].map(lambda x: fmt_or_na(x, 2))

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of the two Uniswap v3 pools used in the empirical study. Volume, swap count, and fee revenue are computed from the same hourly pool data used to build the cleaned price series. Realized volatility is computed from hourly log returns over the retained sample. The current subgraph endpoint returns zeroed WETH/USDC hourly USD volume and fee fields despite nonzero swap counts, so those cells are reported as unavailable.}",
        "\\label{tab:pool-summary}",
        "\\small",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & WETH/USDC & WBTC/USDC \\\\",
        "\\midrule",
    ]

    row_order = [
        "Fee tier",
        "Start date",
        "End date",
        "Hours",
        "Mean hourly volume (USD)",
        "Mean hourly swaps",
        "Hourly realized vol (%)",
        "Annualized realized vol (%)",
        "Mean hourly fee revenue (USD)",
        "Gas / rebalance assumption",
    ]

    left = latex_df.iloc[0]
    right = latex_df.iloc[1]
    for metric in row_order:
        latex_lines.append(f"{metric} & {left[metric]} & {right[metric]} \\\\")

    latex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (OUTPUT_DIR / "pool_summary_table.tex").write_text("\n".join(latex_lines) + "\n")


def main() -> None:
    summary = build_summary()
    write_outputs(summary)
    print(summary.to_string(index=False))
    print(f"\nWrote: {OUTPUT_DIR / 'pool_summary_table.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'pool_summary_table.tex'}")


if __name__ == "__main__":
    main()
