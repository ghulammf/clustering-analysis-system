# import os 
# import pandas as pd
from src.preprocessing.cleaning import DataCleaning

def main():
    dc = DataCleaning()
    dc.load_data()
    dc.select_features()
    dc.handle_missing_values()

if __name__ == "__main__":
    main()