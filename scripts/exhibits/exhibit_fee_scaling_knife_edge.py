import sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np, pandas as pd
from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env
from src.deeprl_liquidity_provision_uniswapv3.data.pools import POOLS
PANEL=R/"data/processed"; W=1500

def run(key, lo, width, fee_mult, cap=30_000.):
    p=POOLS[key]
    df=pd.read_parquet(PANEL/f"{key}_hourly.parquet").iloc[lo:lo+W].reset_index(drop=True).copy()
    df["fees_usd"]=df["fees_usd"]*fee_mult          # scale ONLY fee income
    e=UniswapV3Env(df, fee_tier_pct=p.fee_tier_pct, action_widths=np.array([width]),
                   dec0=p.dec0, dec1=p.dec1, capital_usd=cap, gas_usd=5., warmup=168,
                   allow_exit=False)
    e.reset(); d=tr=False; tot=0.
    while not(d or tr):
        _,r,d,tr,i=e.step(0); tot+=i["reward_true"]
    il=e._hold_usd(e.i)-e._value_usd(e.L,e.i)
    return tot, i["cum_fees"], il

key="usdc_weth_005"; starts=[i*W for i in range(1,9)]
print("Passive LP, held at a fixed width. Fee income scaled; IL untouched.")
print("If IL only dominates because our fees are too small, a modest multiplier")
print("should make the NARROW band win. Where does the argmax move?\n")
print(f"{'fee x':>6} " + "".join(f"{'w='+str(w):>10}" for w in (50,200,500,2000,10000)) + f"{'  argmax':>10} {'fees/IL':>9}")
print("-"*84)
for mult in (1,2,5,10,20,50,100):
    row=[]
    for w in (50,200,500,2000,10000):
        rs=[run(key,s,w,mult) for s in starts]
        row.append(np.median([r[0] for r in rs]))
    # fees/IL at the narrow band, for reference
    rs=[run(key,s,50,mult) for s in starts]
    fi=np.median([r[1] for r in rs])/np.median([r[2] for r in rs])
    best=(50,200,500,2000,10000)[int(np.argmax(row))]
    print(f"{mult:>6}x " + "".join(f"{v:>10,.0f}" for v in row) + f"{best:>10} {fi:>9.2f}")
