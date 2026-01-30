"""
vn30_data_cleaning.py
======================
Quantitative Data Engineering script for cleaning VN30 historical stock data.
Prepares a 'Master DataFrame' for factor scoring and optimization.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats.mstats import winsorize
from xnoapi import client
from xnoapi.vn.data.stocks import Quote
import logging

# Configuration
from config import XNOAPI_KEY, VN30_SYMBOLS, START_DATE, END_DATE

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuantitativeDataEngineer:
    def __init__(self, symbols: List[str], api_key: str):
        self.symbols = symbols
        self.api_key = api_key
        client(apikey=api_key)
        self.raw_data = {}
        self.returns_df = pd.DataFrame()

    def fetch_all_data(self, start: str, end: str):
        """Fetch 10 years of data for all tickers."""
        logging.info(f"Fetching data for {len(self.symbols)} tickers from {start} to {end}")
        for sym in self.symbols:
            try:
                quote = Quote(sym)
                data = quote.history(start=start, interval="1D")
                df = pd.DataFrame(data)
                df['time'] = pd.to_datetime(df['time'])
                df = df.rename(columns={'time': 'date'}).set_index('date')
                df = df[df.index <= end]
                self.raw_data[sym] = df # Store full DataFrame
                logging.info(f"  ✓ {sym} fetched.")
            except Exception as e:
                logging.error(f"  ✗ Failed to fetch {sym}: {e}")

    def clean_and_align(self) -> Dict[str, pd.DataFrame]:
        """
        Main cleaning pipeline resulting in a dictionary of cleaned price DataFrames.
        """
        # 1. Create wide-form Price and Volume DataFrames
        prices_dict = {sym: df['close'] for sym, df in self.raw_data.items()}
        volumes_dict = {sym: df['volume'] for sym, df in self.raw_data.items() if 'volume' in df.columns}
        
        master_price_df = pd.DataFrame(prices_dict)
        master_volume_df = pd.DataFrame(volumes_dict).reindex(master_price_df.index).fillna(0.0)
        logging.info(f"Initial Price DataFrame shape: {master_price_df.shape}")

        # 2. Infinite/NaN Cleanup for Prices
        master_price_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Forward fill then backward fill to maintain price continuity
        master_price_df = master_price_df.ffill().bfill()

        # 3. Consistency & Alignment: Pad early missing dates with the first available price
        # This keeps the returns at 0 for those padded periods
        master_price_df.bfill(inplace=True) 
        logging.info("Step 3: Price alignment and padding completed.")

        # 4. Outlier Mitigation (Winsorization) on Returns
        returns_df = master_price_df.pct_change().fillna(0.0)
        logging.info("Step 4: Applying Winsorization (1st and 99th percentile) on returns...")
        for col in returns_df.columns:
            returns_df[col] = winsorize(returns_df[col], limits=[0.01, 0.01])
            
            # Reconstruct prices to match winsorized returns
            first_price = master_price_df[col].iloc[0]
            master_price_df[col] = first_price * (1 + returns_df[col]).cumprod()

        # 5. Logical Validation
        dropped_tickers = []
        final_cols = []
        for sym in self.symbols:
            if sym not in self.raw_data:
                dropped_tickers.append(sym)
                continue
            orig_prices = self.raw_data[sym]['close']
            missing_pct = orig_prices.isna().sum() / len(master_price_df)
            if missing_pct > 0.10:
                logging.warning(f"  ! Dropping {sym}: {missing_pct:.2%} missing data.")
                dropped_tickers.append(sym)
            else:
                final_cols.append(sym)

        # 6. Reconstruct cleaned dictionary for compatibility with other scripts
        cleaned_data = {}
        for sym in final_cols:
            sym_df = pd.DataFrame({
                'date': master_price_df.index,
                'close': master_price_df[sym].values,
                'daily_return': returns_df[sym].values,
                'volume': master_volume_df[sym].values if sym in master_volume_df.columns else 0.0
            })
            cleaned_data[sym] = sym_df

        logging.info(f"Final pipeline produced cleaned data for {len(cleaned_data)} tickers.")
        return cleaned_data

def main():
    engineer = QuantitativeDataEngineer(VN30_SYMBOLS, XNOAPI_KEY)
    START_DATE_10Y = "2016-01-01"
    END_DATE_NOW = pd.Timestamp.now().strftime("%Y-%m-%d")
    engineer.fetch_all_data(START_DATE_10Y, END_DATE_NOW)
    cleaned_data = engineer.clean_and_align()
    
    # Save a sample Master Returns for verification
    returns_dict = {s: df.set_index('date')['daily_return'] for s, df in cleaned_data.items()}
    master_returns = pd.DataFrame(returns_dict)
    output_path = "output/eda_vn30/data/master_cleaned_returns.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_returns.to_csv(output_path)
    
    logging.info(f"Master Cleaned Returns saved to {output_path}")
    logging.info("\nData Quality Summary:")
    logging.info(master_returns.describe().iloc[:, :5].to_string())

if __name__ == "__main__":
    main()
