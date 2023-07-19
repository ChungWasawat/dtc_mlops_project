

from pathlib import Path
import pandas as pd

def read_data(input: Path):
    return pd.read_csv(input)

def main():
    input_path = Path(__file__).parent.parent.parent
    df = read_data(input_path / "data" / "Fruit Prices 2020.csv")

    print(df.info())
    print(df.head())
    

if __name__ == "__main__":
    main()
