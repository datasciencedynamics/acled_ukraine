################################################################################
######################### Import Requisite Libraries ###########################
import typer
import pandas as pd

################################################################################

from core.constants import var_index

app = typer.Typer()

@app.command()
def main(
    input_data_file: str = "./data/raw/ACLED Data_2026-01-02.csv",
    output_data_file: str = "./data/raw/acled_ukraine_data_2026_01_02.parquet"
):
    """
    Main script execution replacing sys.argv with typer.

    Args:
        input_data_file (str): Path to the input csv file.
        output_data_file (str): Path to save the processed parquet file.
    """
    ############################################################################
    # Step 1. Read the input data file
    ############################################################################

    # read in the data and set the index as 'event_id_cnty'
    df = pd.read_csv(input_data_file)
    try:
        df.set_index(var_index, inplace=True)
    except:
        print("Index already set or 'var_index' doesn't exist in dataframe")

    print(f"Input Data Shape: {df.shape}")
    print(f"There are {df.index.unique().shape[0]} unique indices in the dataframe.")

    df.to_parquet(output_data_file)

if __name__ == "__main__":
    app()
