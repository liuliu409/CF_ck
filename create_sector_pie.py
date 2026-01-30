import matplotlib.pyplot as plt
import pandas as pd
import os

# Data from config.py
STOCK_SECTORS = {
    'ACB': 'Banking', 'BID': 'Banking', 'CTG': 'Banking', 'HDB': 'Banking',
    'MBB': 'Banking', 'SHB': 'Banking', 'SSB': 'Banking', 'STB': 'Banking',
    'TCB': 'Banking', 'TPB': 'Banking', 'VCB': 'Banking', 'VIB': 'Banking',
    'VPB': 'Banking',
    'VHM': 'Real Estate', 'VIC': 'Real Estate', 'VRE': 'Real Estate',
    'MSN': 'Consumer', 'MWG': 'Consumer', 'SAB': 'Consumer', 'VJC': 'Consumer',
    'VNM': 'Consumer',
    'BCM': 'Industrial', 'GVR': 'Industrial', 'HPG': 'Industrial',
    'GAS': 'Utilities', 'PLX': 'Utilities', 'POW': 'Utilities',
    'FPT': 'Technology',
    'BVH': 'Financial', 'SSI': 'Financial'
}

def create_pie_chart():
    # Count sectors
    df = pd.Series(STOCK_SECTORS).value_counts()
    
    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Modern color palette
    colors = plt.cm.get_cmap('Set3')(range(len(df)))
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(
        df, 
        labels=df.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        explode=[0.05 if x == 'Banking' else 0 for x in df.index], # Explode the largest sector
        shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    plt.title('VN30 Sector Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Layout adjustment
    plt.tight_layout()
    
    # Ensure output directory exists
    output_path = "output/eda_vn30/figures/sector_distribution.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    create_pie_chart()
