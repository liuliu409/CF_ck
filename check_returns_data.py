import pandas as pd
import numpy as np
from xnoapi import client
from xnoapi.vn.data.stocks import Quote
from config import XNOAPI_KEY, VN30_SYMBOLS, START_DATE

client(apikey=XNOAPI_KEY)

def check_data():
    all_returns = []
    for sym in VN30_SYMBOLS[:5]: # Check first 5
        try:
            quote = Quote(sym)
            data = quote.history(start=START_DATE, interval="1D")
            df = pd.DataFrame(data)
            if not df.empty and 'close' in df.columns:
                returns = df['close'].pct_change().dropna()
                all_returns.append(returns)
                print(f"{sym}: count={len(returns)}, mean={returns.mean():.6f}, std={returns.std():.6f}, min={returns.min():.6f}, max={returns.max():.6f}")
        except Exception as e:
            print(f"Error {sym}: {e}")
    
    if all_returns:
        combined = pd.concat(all_returns)
        print("\nCombined Stats:")
        print(combined.describe())
        print(f"Kurtosis: {combined.kurtosis()}")
        print(f"Skewness: {combined.skewness() if hasattr(combined, 'skewness') else combined.skew()}")

if __name__ == "__main__":
    check_data()
