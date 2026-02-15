################################################################################
######################### Import Requisite Libraries ###########################
from pathlib import Path
import pandas as pd
import numpy as np
import typer
from sentence_transformers import SentenceTransformer

################################################################################

app = typer.Typer()


@app.command()
def main(
    input_data_file: str = "./data/processed/text_base.parquet",
    output_data_file: str = "./data/processed/text_embeddings.parquet",
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """
    Generate sentence embeddings from cleaned text.

    Args:
        input_data_file (str): Path to cleaned text parquet file.
        output_data_file (str): Path to output embeddings parquet file.
        model_name (str): Sentence transformer model to use.
    """

    input_path = Path(input_data_file)
    output_path = Path(output_data_file)

    ############################################################################
    # Step 1. Read Cleaned Text Data
    ############################################################################

    print("Starting text embedding pipeline...")
    print(f"Reading cleaned text from: {input_path}")

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows")

    ############################################################################
    # Step 2. Load Sentence Transformer Model
    ############################################################################

    print()
    print(f"Loading sentence transformer model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {embedder.get_sentence_embedding_dimension()}")

    ############################################################################
    # Step 3. Generate Embeddings
    ############################################################################

    print()
    print("Generating embeddings...")
    embeddings = embedder.encode(
        df['notes_clean_ml'].tolist(),
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    print(f"Embeddings generated. Shape: {embeddings.shape}")

    ############################################################################
    # Step 4. Build Output DataFrame
    ############################################################################

    print()
    print("Building output DataFrame...")

    # Create column names
    n_dims = embeddings.shape[1]
    col_names = [f'text_emb_{i}' for i in range(n_dims)]

    # Create embeddings DataFrame
    emb_df = pd.DataFrame(embeddings, columns=col_names)
    emb_df['event_id_cnty'] = df['event_id_cnty'].values

    # Reorder columns to have event_id_cnty first
    cols = ['event_id_cnty'] + col_names
    emb_df = emb_df[cols]

    ############################################################################
    # Step 5. Save Embeddings
    ############################################################################

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Saving to: {output_path}")
    emb_df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"  Final shape: {len(emb_df):,} rows x {emb_df.shape[1]} column(s)")
    print()
    print("Embedding complete!")


if __name__ == "__main__":
    app()