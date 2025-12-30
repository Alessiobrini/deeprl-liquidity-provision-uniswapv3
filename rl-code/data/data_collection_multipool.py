"""
Multi-Pool Data Collection for Uniswap v3
==========================================

This script collects historical hourly pool data from Uniswap v3 via The Graph subgraph API.
Uses poolHourDatas query for efficient hourly data retrieval with pre-computed prices.

Usage:
    python data_collection_multipool.py --pool wbtc_usdc_030
    python data_collection_multipool.py --pool weth_usdc_005
    python data_collection_multipool.py --pool all
"""

import requests
import pandas as pd
import numpy as np
import time
import argparse
import os
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Pool configurations
# Reference: https://thegraph.com/explorer/subgraphs/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV
POOL_CONFIGS = {
    'weth_usdc_005': {
        'address': '0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640',
        'name': 'WETH/USDC',
        'token0': 'USDC',  
        'token1': 'WETH',
        'fee_tier': 0.05,
        'tick_spacing': 10,
        'price_field': 'token1Price',  # token1Price = WETH price in USDC terms
        'output_file': 'data_price_weth_usdc_005_h_time.csv',
        'description': 'WETH/USDC 0.05% - Highest volume pool'
    },
    'wbtc_usdc_030': {
        'address': '0x99ac8cA7087fa4A2A1FB6357269965A2014ABc35',
        'name': 'WBTC/USDC',
        'token0': 'WBTC',  
        'token1': 'USDC',
        'fee_tier': 0.30,
        'tick_spacing': 60,
        'price_field': 'token0Price',  # token0Price = WBTC price in USDC terms
        'invert_price': True,  # Subgraph returns inverse ratio for this pool
        'output_file': 'data_price_wbtc_usdc_030_h_time.csv',
        'description': 'WBTC/USDC 0.3% - Top 10 pool by TVL'
    }
}


def unix_epoch_to_timestamp(unix_epochs, unit='s'):
    """Convert Unix epoch times to human-readable timestamps."""
    if not isinstance(unix_epochs, pd.Series):
        unix_epochs = pd.Series(unix_epochs)
    unix_epochs_numeric = pd.to_numeric(unix_epochs, errors='coerce')
    return pd.to_datetime(unix_epochs_numeric, unit=unit)


def verify_pool_config(pool_address, subgraph_url=None):
    """
    Verify pool configuration by querying the subgraph for pool details.

    Returns pool info including token0, token1, feeTier to validate our config.
    """
    if subgraph_url is None:
        # Load API key from environment variable
        API_KEY = os.getenv('THEGRAPH_API_KEY')
        if not API_KEY:
            raise ValueError(
                "THEGRAPH_API_KEY not found in environment. "
                "Please create a .env file with your API key. "
                "See .env.example for template."
            )
        subgraph_url = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

    query = """
    {
      pool(id: "%s") {
        token0 {
          symbol
          decimals
        }
        token1 {
          symbol
          decimals
        }
        feeTier
        liquidity
        sqrtPrice
        tick
      }
    }
    """ % pool_address.lower()

    try:
        response = requests.post(subgraph_url, json={'query': query}, timeout=10)
        data = response.json()
    except Exception as e:
        raise ValueError(f"Failed to query subgraph: {e}")

    # Debug: print response
    if 'errors' in data:
        raise ValueError(f"Subgraph errors: {data['errors']}")

    if not data.get('data'):
        raise ValueError(f"No data in response: {data}")

    pool_info = data.get('data', {}).get('pool')
    if not pool_info:
        raise ValueError(f"Pool {pool_address} not found. Response: {data}")

    return {
        'token0_symbol': pool_info['token0']['symbol'],
        'token1_symbol': pool_info['token1']['symbol'],
        'token0_decimals': int(pool_info['token0']['decimals']),
        'token1_decimals': int(pool_info['token1']['decimals']),
        'fee_tier': int(pool_info['feeTier']) / 10000,  # Convert basis points to percentage
        'liquidity': pool_info['liquidity']
    }


def fetch_hourly_pool_data(pool_address, last_timestamp=0, subgraph_url=None):
    """
    Fetch hourly pool data from Uniswap v3 subgraph.

    Uses poolHourDatas which provides pre-aggregated hourly metrics including:
    - periodStartUnix: hour timestamp
    - token0Price: price of token0 in terms of token1
    - token1Price: price of token1 in terms of token0
    - liquidity, volumeUSD, feesUSD, tvlUSD, txCount
    """
    if subgraph_url is None:
        # Load API key from environment variable
        API_KEY = os.getenv('THEGRAPH_API_KEY')
        if not API_KEY:
            raise ValueError(
                "THEGRAPH_API_KEY not found in environment. "
                "Please create a .env file with your API key."
            )
        subgraph_url = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

    query = """
    {
      poolHourDatas(
        first: 1000,
        where: {pool: "%s", periodStartUnix_gt: %d},
        orderBy: periodStartUnix,
        orderDirection: asc
      ) {
        periodStartUnix
        liquidity
        sqrtPrice
        token0Price
        token1Price
        volumeUSD
        feesUSD
        tvlUSD
        txCount
        open
        high
        low
        close
      }
    }
    """ % (pool_address.lower(), last_timestamp)

    response = requests.post(subgraph_url, json={'query': query})
    return response


def collect_pool_hourly_data(pool_config, max_retries=3):
    """
    Collect all historical hourly pool data.

    This is much more efficient than collecting individual swaps since
    the subgraph pre-aggregates data into hourly buckets.
    """
    pool_address = pool_config['address']
    print(f"\n{'='*70}")
    print(f"Collecting hourly data for {pool_config['name']} ({pool_config['fee_tier']}%)")
    print(f"{pool_config['description']}")
    print(f"Pool Address: {pool_address}")
    print(f"{'='*70}\n")

    # First, verify pool configuration
    print("Verifying pool configuration...")
    try:
        pool_info = verify_pool_config(pool_address)
        print(f"[OK] Pool verified:")
        print(f"  Token0: {pool_info['token0_symbol']} ({pool_info['token0_decimals']} decimals)")
        print(f"  Token1: {pool_info['token1_symbol']} ({pool_info['token1_decimals']} decimals)")
        print(f"  Fee Tier: {pool_info['fee_tier']}%")
        print(f"  Current Liquidity: {float(pool_info['liquidity']):,.0f}")

        # Update config with verified info if different
        if pool_config['token0'] != pool_info['token0_symbol']:
            print(f"\n[WARNING] Config has token0={pool_config['token0']}, "
                  f"but subgraph shows {pool_info['token0_symbol']}")
            print(f"  Updating configuration to match subgraph...")
            pool_config['token0'] = pool_info['token0_symbol']
            pool_config['token1'] = pool_info['token1_symbol']

    except Exception as e:
        print(f"[ERROR] Could not verify pool: {e}")
        print("  Proceeding with configured values...")

    # Collect hourly data
    all_hours = []
    last_timestamp = 0

    with tqdm(desc="Fetching hourly data", unit=" batches") as pbar:
        while True:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    response = fetch_hourly_pool_data(pool_address, last_timestamp)
                    data = response.json()

                    if 'errors' in data:
                        print(f"\nAPI Error: {data['errors']}")
                        retry_count += 1
                        time.sleep(2 ** retry_count)
                        continue

                    hours = data.get("data", {}).get("poolHourDatas", [])
                    break

                except Exception as e:
                    print(f"\nRequest failed: {e}")
                    retry_count += 1
                    time.sleep(2 ** retry_count)

            if retry_count >= max_retries:
                print(f"\nFailed after {max_retries} retries. Stopping...")
                break

            if not hours:
                print("\nNo more data found. Collection complete!")
                break

            all_hours.extend(hours)
            last_timestamp = int(hours[-1]["periodStartUnix"])
            last_time_readable = unix_epoch_to_timestamp(last_timestamp).item()

            pbar.update(1)
            pbar.set_postfix({
                'Total Hours': len(all_hours),
                'Last Time': last_time_readable.strftime('%Y-%m-%d %H:%M')
            })

            # Respect rate limits
            time.sleep(0.3)

    print(f"\n[OK] Total hours collected: {len(all_hours):,}")
    return pd.DataFrame(all_hours)


def process_hourly_data(df, pool_config):
    """
    Process hourly pool data to extract prices.

    The subgraph provides:
    - token0Price: price of token0 in terms of token1
    - token1Price: price of token1 in terms of token0

    We want price in USDC terms:
    - For WETH/USDC (token0=USDC, token1=WETH): use token1Price (USDC per WETH)
    - For WBTC/USDC (token0=WBTC, token1=USDC): use token0Price (USDC per WBTC)
    """
    print("\nProcessing hourly data...")

    # Convert timestamp
    df['timestamp'] = df['periodStartUnix'].apply(lambda x: unix_epoch_to_timestamp(int(x)))
    df = df.set_index('timestamp')

    # Convert price fields to float
    df['token0Price'] = pd.to_numeric(df['token0Price'], errors='coerce')
    df['token1Price'] = pd.to_numeric(df['token1Price'], errors='coerce')

    # Use the price field specified in config (already verified)
    price_column = pool_config['price_field']
    df['price'] = df[price_column]

    # Invert if needed (some pools return inverse ratios)
    if pool_config.get('invert_price', False):
        df['price'] = 1.0 / df['price']
        print(f"  Inverting {price_column} to get correct price")

    # Log which price we're using
    if price_column == 'token0Price':
        print(f"  Using token0Price: {pool_config['token0']} price in {pool_config['token1']} terms")
    else:
        print(f"  Using token1Price: {pool_config['token1']} price in {pool_config['token0']} terms")

    # Basic stats
    print(f"[OK] Processed {len(df):,} hours of data")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"  Mean price: ${df['price'].mean():.2f}")
    print(f"  Std dev: ${df['price'].std():.2f}")

    return df


def clean_hourly_data(df, pool_config):
    """
    Clean hourly price data:
    - Remove NaN/zero prices
    - Forward fill missing hours
    - Remove outliers (extreme prices and returns)
    """
    print("\nCleaning data...")

    original_len = len(df)

    # Get price series
    price = df['price'].copy()

    # Replace zeros with NaN
    price = price.replace(0.0, np.nan)
    nan_count = price.isna().sum()
    if nan_count > 0:
        print(f"  Found {nan_count} NaN/zero prices ({100*nan_count/len(price):.1f}%)")

    # Create complete hourly index
    full_range = pd.date_range(start=price.index.min(), end=price.index.max(), freq='h')
    price = price.reindex(full_range)
    missing_hours = price.isna().sum()
    if missing_hours > nan_count:
        print(f"  Found {missing_hours - nan_count} missing hours in sequence")

    # Forward fill
    price = price.ffill()

    # Remove extreme prices (> 5x median or < 0.2x median)
    median_price = price.median()
    extreme_high = price > (5 * median_price)
    extreme_low = price < (0.2 * median_price)
    extreme_mask = extreme_high | extreme_low

    if extreme_mask.sum() > 0:
        print(f"  Removing {extreme_mask.sum()} extreme prices")
        print(f"    - {extreme_high.sum()} above 5x median (${5*median_price:.2f})")
        print(f"    - {extreme_low.sum()} below 0.2x median (${0.2*median_price:.2f})")
        price[extreme_mask] = np.nan

    # Remove extreme returns (> ±20% hourly)
    returns = price.pct_change(fill_method=None)
    extreme_returns = (returns > 0.2) | (returns < -0.2)

    if extreme_returns.sum() > 0:
        print(f"  Removing {extreme_returns.sum()} hours with extreme returns (>±20%)")
        price[extreme_returns] = np.nan

    # Forward fill after cleaning
    price = price.ffill()

    # Drop remaining NaN (at beginning)
    price = price.dropna()

    cleaned_len = len(price)
    print(f"[OK] Cleaning complete: {original_len:,} -> {cleaned_len:,} hours "
          f"({100*cleaned_len/original_len:.1f}% retained)")

    return price


def save_hourly_data(hourly_price, output_path):
    """
    Save hourly price data to CSV in the format expected by training scripts.

    Format:
    timestamp,price
    5/5/2021 1:00,3296.08
    5/5/2021 2:00,3296.08
    """
    print(f"\nSaving data to {output_path}...")

    # Convert to DataFrame
    df_output = hourly_price.reset_index()
    df_output.columns = ['timestamp', 'price']

    # Format timestamp to match existing format: "M/D/YYYY H:MM"
    # Note: Windows doesn't support %-m, use %#m instead
    try:
        df_output['timestamp'] = df_output['timestamp'].dt.strftime('%-m/%-d/%Y %-H:%M')
    except:
        # Windows format
        df_output['timestamp'] = df_output['timestamp'].dt.strftime('%#m/%#d/%Y %#H:%M')

    # Save to CSV
    df_output.to_csv(output_path, index=False)

    file_size = Path(output_path).stat().st_size / 1024
    print(f"[OK] Data saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {file_size:.1f} KB")
    print(f"  Rows: {len(df_output):,}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Collect Uniswap v3 hourly pool data for RL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect WBTC/USDC data
  python data_collection_multipool.py --pool wbtc_usdc_030

  # Collect WETH/USDC data
  python data_collection_multipool.py --pool weth_usdc_005

  # Collect both pools
  python data_collection_multipool.py --pool all

  # Save to specific directory
  python data_collection_multipool.py --pool all --output-dir ./data
        """
    )

    parser.add_argument(
        '--pool',
        type=str,
        choices=list(POOL_CONFIGS.keys()) + ['all'],
        required=True,
        help='Pool to collect data for (or "all" for all pools)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for CSV files (default: current directory)'
    )

    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify pool configurations, do not collect data'
    )

    args = parser.parse_args()

    # Determine which pools to process
    if args.pool == 'all':
        pools_to_process = POOL_CONFIGS.keys()
    else:
        pools_to_process = [args.pool]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify-only mode
    if args.verify_only:
        print("\n" + "="*70)
        print("POOL VERIFICATION MODE")
        print("="*70)
        for pool_name in pools_to_process:
            pool_config = POOL_CONFIGS[pool_name]
            try:
                pool_info = verify_pool_config(pool_config['address'])
                print(f"\n[OK] {pool_name}:")
                print(f"  Address: {pool_config['address']}")
                print(f"  Token0: {pool_info['token0_symbol']} ({pool_info['token0_decimals']} decimals)")
                print(f"  Token1: {pool_info['token1_symbol']} ({pool_info['token1_decimals']} decimals)")
                print(f"  Fee Tier: {pool_info['fee_tier']}%")
            except Exception as e:
                print(f"\n[X] {pool_name}: {e}")
        return

    # Process each pool
    results = {}
    for pool_name in pools_to_process:
        pool_config = POOL_CONFIGS[pool_name]

        try:
            # Collect hourly data
            hourly_df = collect_pool_hourly_data(pool_config)

            if len(hourly_df) == 0:
                print(f"\n[X] No data collected for {pool_name}. Skipping...")
                results[pool_name] = 'FAILED: No data'
                continue

            # Process to get prices
            processed_df = process_hourly_data(hourly_df, pool_config)

            # Clean data
            cleaned_price = clean_hourly_data(processed_df, pool_config)

            # Save to CSV
            output_path = output_dir / pool_config['output_file']
            save_hourly_data(cleaned_price, output_path)

            results[pool_name] = f'SUCCESS: {len(cleaned_price):,} hours'

            print(f"\n{'='*70}")
            print(f"[OK] {pool_name} processing complete!")
            print(f"{'='*70}\n")

        except Exception as e:
            print(f"\n[X] Error processing {pool_name}: {e}")
            import traceback
            traceback.print_exc()
            results[pool_name] = f'FAILED: {str(e)}'
            continue

    # Final summary
    print("\n" + "="*70)
    print("DATA COLLECTION SUMMARY")
    print("="*70)
    for pool_name, result in results.items():
        status_symbol = "[OK]" if "SUCCESS" in result else "[X]"
        print(f"{status_symbol} {pool_name}: {result}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
