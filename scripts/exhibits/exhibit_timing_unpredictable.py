import sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np, pandas as pd
from src.deeprl_liquidity_provision_uniswapv3.envs.uniswap_v3 import UniswapV3Env
from src.deeprl_liquidity_provision_uniswapv3.data.pools import POOLS
PANEL=R/"data/processed"; W=1500; WARMUP=168

def reward(key, lo, width=200.):
    p=POOLS[key]; df=pd.read_parquet(PANEL/f"{key}_hourly.parquet").iloc[lo:lo+W].reset_index(drop=True)
    if len(df)<W: return None
    e=UniswapV3Env(df, fee_tier_pct=p.fee_tier_pct, action_widths=np.array([width]),
                   dec0=p.dec0, dec1=p.dec1, capital_usd=30_000., gas_usd=5., warmup=WARMUP)
    e.reset(); e.step(2)  # enter, then hold for the window
    t=0.; d=tr=False
    while not(d or tr):
        _,r,d,tr,_=e.step(0); t+=r
    return t

print("Do PRE-window observables predict the window's REALISED reward?")
print("y = reward of entering at w=200 and holding. x = features from the PRIOR window.")
print("This is the question. My earlier test regressed the signal on itself and")
print("measured its autocorrelation instead.\n")
print(f"{'pool':<15} {'n':>4} {'corr(x,y)':>10} {'sign acc':>9} {'oracle':>9} {'rule':>9} {'always-in':>10}")
print("-"*72)
for key in ["usdc_weth_005","usdc_weth_030","wbtc_weth_005","wbtc_weth_030","weth_usdt_005","weth_usdt_030"]:
    df=pd.read_parquet(PANEL/f"{key}_hourly.parquet")
    n=len(df)//W
    ys=np.array([reward(key,i*W) for i in range(n)],dtype=float)
    xs=[]
    for i in range(n):
        a,b=max(0,(i-1)*W),i*W          # strictly the PRIOR window
        if i==0: xs.append(np.nan); continue
        pay=df.fees_usd.values[a:b].sum()/max(df.liquidity.values[a:b].mean(),1e-30)
        var=np.var(np.diff(np.log(np.maximum(df.price.values[a:b],1e-12))))
        xs.append(np.log(pay/max(var,1e-30)+1e-30))
    xs=np.array(xs)
    m=~np.isnan(xs)&~np.isnan(ys)
    x,y=xs[m],ys[m]
    c=np.corrcoef(x,y)[0,1]
    thr=np.median(x)
    acc=((x>thr)==(y>0)).mean()
    oracle=np.mean(np.maximum(y,0))
    rule=np.mean(np.where(x>thr,y,0))    # in when signal rich, else out
    print(f"{key:<15} {len(y):>4} {c:>10.2f} {acc:>8.0%} {oracle:>9,.0f} {rule:>9,.0f} {np.mean(y):>10,.0f}")
