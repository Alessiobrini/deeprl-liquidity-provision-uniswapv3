import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Load WBT C data
wbtc_df = pd.read_csv('data_price_wbtc_usdc_030_h_time.csv')
wbtc_df['timestamp'] = pd.to_datetime(wbtc_df['timestamp'])

print(f"WBTC/USDC ({len(wbtc_df):,} hours):")
print(f"  Date: {wbtc_df.timestamp.min()} to {wbtc_df.timestamp.max()}")
print(f"  Price: ${wbtc_df.price.min():.2f} to ${wbtc_df.price.max():.2f}")
print(f"  Mean: ${wbtc_df.price.mean():.2f}, Median: ${wbtc_df.price.median():.2f}")

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(wbtc_df.timestamp, wbtc_df.price, linewidth=0.5, alpha=0.8, color='darkblue')
ax.set_title('WBTC/USDC Price (Uniswap v3 0.3% Pool)', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USDC per WBTC)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('wbtc_usdc_price.png', dpi=150, bbox_inches='tight')
print("\nSaved: wbtc_usdc_price.png")
plt.close()

# Compare with WETH if available
try:
    weth_df = pd.read_csv('../data_price_uni_h_time.csv')
    weth_df['timestamp'] = pd.to_datetime(weth_df['timestamp'])
    print(f"\nWETH/USDC ({len(weth_df):,} hours):")
    print(f"  Date: {weth_df.timestamp.min()} to {weth_df.timestamp.max()}")
    print(f"  Price: ${weth_df.price.min():.2f} to ${weth_df.price.max():.2f}")
    print(f"  Mean: ${weth_df.price.mean():.2f}, Median: ${weth_df.price.median():.2f}")

    # Ratio plot
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(weth_df.timestamp, weth_df.price, label='WETH', linewidth=0.5, alpha=0.7, color='blue')
    ax1.set_title('WETH/USDC Price', fontsize=12, fontweight='bold')
    ax1.set_ylabel('USDC per WETH')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(wbtc_df.timestamp, wbtc_df.price, label='WBTC', linewidth=0.5, alpha=0.7, color='orange')
    ax2.set_title('WBTC/USDC Price', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('USDC per WBTC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('pool_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: pool_comparison.png")
    plt.close()

except FileNotFoundError:
    print("\nWETH/USDC data not found")
