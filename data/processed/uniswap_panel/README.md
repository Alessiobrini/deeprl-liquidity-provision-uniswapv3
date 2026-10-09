# Processed panel

Hourly panel and per-swap fees for the six core Uniswap v3 pools, aggregated from public Swap events on Ethereum mainnet.

    <pool>_hourly.parquet   one row per hour, complete grid, May 2021 to June 2026
    <pool>_swaps.parquet    one row per swap: its price interval, its active
                            liquidity, and the fee it paid

The files are too large for GitHub (about 460 MB) and are available on request from Alessio Brini (alessio.brini@duke.edu) or Arjun Mahesh (arjun.mahesh@duke.edu). Place them in this folder, which is all the code reads. The main README explains how to rebuild them from raw Swap events.
