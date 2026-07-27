# Panel bundle

Hourly panel and swap-level fees for the six core Uniswap v3 pools, aggregated from
raw Swap events on Ethereum mainnet.

    <pool>_hourly.parquet   one row per hour, complete grid, 2021-05 to 2026-06
    <pool>_swaps.parquet    one row per swap: its price interval, its active
                            liquidity, and the fee it paid

Unzip into `data/processed/` at the repo root. Nothing else is needed; the code reads
only these files.

Regenerating them needs the raw swap parquet from the sibling `defi-rv` project
(~4.9GB, not distributed here):

    python -m src.deeprl_liquidity_provision_uniswapv3.data.aggregate
    python -m src.deeprl_liquidity_provision_uniswapv3.data.swaps

Columns are documented in `data/DATA_DICTIONARY.md`.
