
import yfinance as yf
import pandas as pd

try:
    stock = yf.Ticker("2385.TW")
    print("Fetching financials...")
    fin = stock.quarterly_financials
    print(f"Financials shape: {fin.shape}")
    if not fin.empty:
        print(fin.head())
        if "Total Revenue" in fin.index:
            print("Total Revenue found.")
        else:
            print("Total Revenue NOT found in index.")
            print(fin.index)
    else:
        print("Financials empty.")
        
    print("-" * 20)
    print("Fetching dividends...")
    div = stock.dividends
    print(f"Dividends shape: {div.shape}")
    if not div.empty:
        print(div.tail())
    else:
        print("Dividends empty.")

except Exception as e:
    print(f"Error: {e}")
