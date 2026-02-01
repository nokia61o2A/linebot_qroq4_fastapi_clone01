
import pandas as pd

def load_stock_data():
    return pd.read_csv('name_df.csv')

def get_stock_name(stock_id):
    stock_data_df = load_stock_data()
    # Print types for debugging
    print(f"Input stock_id: {stock_id} (type: {type(stock_id)})")
    print(f"DataFrame '股號' type: {stock_data_df['股號'].dtype}")
    
    # Check if stock_id exists
    # Try direct comparison
    result = stock_data_df[stock_data_df['股號'] == stock_id]
    print(f"Direct comparison result empty: {result.empty}")
    
    # Try string conversion comparison
    stock_data_df['股號_str'] = stock_data_df['股號'].astype(str)
    result_str = stock_data_df[stock_data_df['股號_str'] == str(stock_id)]
    print(f"String comparison result empty: {result_str.empty}")
    
    if not result.empty:
        return result.iloc[0]['股名']
    if not result_str.empty:
        return result_str.iloc[0]['股名']
        
    return None

if __name__ == "__main__":
    name = get_stock_name("2385")
    print(f"Result for 2385: {name}")
    
    name_int = get_stock_name(2385)
    print(f"Result for 2385 (int): {name_int}")
