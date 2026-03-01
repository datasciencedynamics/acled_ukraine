################################################################################
######################### Import Requisite Libraries ###########################
import os
import typer
import pandas as pd
import numpy as np

# from node2vec import Node2Vec  # COMMENTED OUT: No longer using embeddings

# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

################################################################################

from core.constants import (
    var_index,
    preproc_run_name,
    exp_artifact_name,
    percent_miss,
    drop_vars,
)

# import all user-defined functions and constants
from core.functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    haversine_km,
    safe_to_numeric,
    # build_actor_interaction_graph,  # COMMENTED OUT: No longer using embeddings
    # add_pairwise_embedding_features,  # COMMENTED OUT: No longer using embeddings
    # apply_embeddings,  # COMMENTED OUT: No longer using embeddings
)

app = typer.Typer()

print("\n" + "#" * 80)
print(f"Running script: {os.path.basename(__file__)}")
print("#" * 80 + "\n")


@app.command()
def main(
    input_data_file: str = "./data/raw/acled_ukraine_data_2026_01_02.parquet",
    output_data_file: str = "./data/processed/df_sans_zero_missing.parquet",
    stage: str = "training",
    data_path: str = "./data/processed",
):
    """
    Main script execution replacing sys.argv with typer.

    Args:
        input_data_file (str): Path to the input parquet file.
        output_data_file (str): Path to save the processed parquet file.
        stage (str): Processing stage (e.g., 'training' or 'inference').
    """

    ############################################################################
    # Step 1. Read the input data file
    ############################################################################
    # IMPORTANT UPDATE:
    # We do NOT build embeddings from the full raw dataframe anymore.
    # We load the temporal splits created by temporal_splits.py, then:
    #   1) Build the actor graph ONLY from train_df (prevents temporal leakage)
    #   2) Train Node2Vec embeddings on that training graph
    #   3) Apply the learned embeddings to train, valid, and test splits
    #
    # UPDATE 2026-02-16: EMBEDDINGS COMMENTED OUT FOR INTERPRETABILITY
    ############################################################################

    train_path = os.path.join(data_path, "train_df.parquet")
    valid_path = os.path.join(data_path, "valid_df.parquet")
    test_path = os.path.join(data_path, "test_df.parquet")

    if (
        os.path.exists(train_path)
        and os.path.exists(valid_path)
        and os.path.exists(test_path)
    ):
        print("Loading temporal splits...")
        train_df = pd.read_parquet(train_path)
        valid_df = pd.read_parquet(valid_path)
        test_df = pd.read_parquet(test_path)
    else:
        raise FileNotFoundError(
            "Temporal splits not found. Run temporal_splits.py first "
            "to create train_df.parquet, valid_df.parquet, test_df.parquet."
        )

    # Set index if needed (on each split)
    for _name, _df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        try:
            _df.set_index(var_index, inplace=True)
        except Exception:
            print(
                f"({_name}) Index already set or '{var_index}' doesn't exist in dataframe"
            )

    print(f"Train Data Shape: {train_df.shape}")
    print(f"Valid Data Shape: {valid_df.shape}")
    print(f"Test Data Shape:  {test_df.shape}")

    print(
        f"There are {train_df.index.unique().shape[0]} unique indices in the TRAIN dataframe."
    )

    ############################################################################
    # Step 2. String Columns Handling
    ############################################################################
    # String columns are identified and should be removed before modeling
    # because machine learning models typically require numerical inputs.
    # Keeping string columns in the dataset may lead to errors or
    # unintended behavior unless explicitly encoded.
    #
    # To ensure consistency between training and inference,
    # we save the list of string columns and track it using MLflow.
    ############################################################################

    if stage == "training":

        # Identify string columns on TRAIN only
        df_object = train_df.select_dtypes("object")
        print()
        print(
            "The following columns have strings and may need to be removed from "
            "modeling and/or otherwise transformed with `categorical_transformer` "
            f"\nas handled accordingly in the `config.py` file. This list is stored "
            f"as an artifact in MLflow for future reference if necessary for "
            f"retrieval at a later time. \n \n"
            f"There are {df_object.shape[1]} string columns:\n \n"
            f"{df_object.columns.to_list()}. \n "
        )

        # Extract column names to a list
        string_cols_list = df_object.columns.to_list()

        ############################################################################
        # Step 3. Save and Log String Column List
        ############################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        # This list of string columns is dumped (stored) only to inform of what
        # the string columns are; no further action is taken; we do not need to
        # load this list into production, since it is only there for us to
        # see what the columns are.
        ############################################################################

        # Dump the string_cols_list into a pickle file for future reference
        dumpObjects(
            string_cols_list,
            os.path.join(data_path, "string_cols_list.pkl"),
        )

        # Log the string column list as an artifact in MLflow
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="string_cols_list",
            obj=string_cols_list,
        )

        ############################################################################
        # Step 4. Store Unique Actor Lists
        ############################################################################
        # Extract unique actor1 and actor2 lists from the dataset and store them
        # as pickle files for future reference. Additionally, log these lists as
        # artifacts in MLflow to ensure reproducibility and easy access during
        # model training and inference.
        #
        # IMPORTANT UPDATE:
        # Store lists from TRAIN only to avoid accidentally recording future-only
        # actors as part of the "known" set.
        ############################################################################

        actor1_list = train_df["actor1"].unique().tolist()
        actor2_list = train_df["actor2"].unique().tolist()

        dumpObjects(actor1_list, os.path.join(data_path, "actor1_list.pkl"))
        dumpObjects(actor2_list, os.path.join(data_path, "actor2_list.pkl"))

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor1_list",
            obj=actor1_list,
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor2_list",
            obj=actor2_list,
        )

    if stage == "inference":

        ########################################################################
        # Load Previously Saved Object
        ########################################################################
        # During training, we identified and stored actor lists.
        # Now, we reload this to ensure that inference follows the same
        # preprocessing pipeline as training, maintaining consistency.
        ########################################################################

        actor1_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor1_list",
        )
        actor2_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor2_list",
        )

    ############################################################################
    ###################### Re-engineering Selected Features ####################
    ############################################################################

    ############################################################################
    # Step 5. Normalize Actor Names and Create Missing Indicator
    ############################################################################
    # Normalize actor names in 'actor1' and 'actor2' columns to ensure
    # consistency and reduce variability caused by different naming conventions.
    # This step helps in better analysis and modeling by standardizing the
    # representation of actors involved in events.
    #
    # IMPORTANT UPDATE:
    # We normalize EACH SPLIT independently but using the same normalize_split()
    # logic to keep the pipeline consistent.
    ############################################################################

    # train_df = normalize_split(train_df)
    # valid_df = normalize_split(valid_df)
    # test_df = normalize_split(test_df)

    # print(
    #     f"\nTop 10 Normalized Actor 1 Names (TRAIN): \n{train_df['actor1_root'].value_counts().head(10)}"
    # )
    # print(
    #     f"\nTop 10 Normalized Actor 2 Names (TRAIN): \n{train_df['actor2_root'].value_counts().head(10)}"
    # )

    # ############################################################################
    # # Step 6. Build Actor Interaction Graph
    # ############################################################################
    # # COMMENTED OUT 2026-02-16: Removing embeddings for interpretability
    # # Construct a graph representing interactions between actors based on
    # # co-occurrences in events. This graph captures relationships and patterns
    # # among actors, which can be useful for network analysis and feature
    # # engineering in predictive modeling.
    # #
    # # IMPORTANT UPDATE:
    # # Build the interaction graph ONLY from TRAIN data to prevent temporal leakage.
    # ############################################################################
    #
    # G = build_actor_interaction_graph(train_df)
    #
    # ############################################################################
    # # Step 7. Generate Node2Vec Embeddings
    # ############################################################################
    # # COMMENTED OUT 2026-02-16: Removing embeddings for interpretability
    # # Generate Node2Vec embeddings for actors in the interaction graph.
    # # These embeddings capture the structural relationships and properties
    # # of actors within the network, providing valuable features for downstream
    # # machine learning tasks.
    # ############################################################################
    #
    # node2vec = Node2Vec(
    #     G,
    #     dimensions=32,
    #     walk_length=10,
    #     num_walks=50,
    #     workers=1,
    #     weight_key="weight",
    #     seed=seed,
    # )
    #
    # model = node2vec.fit(window=5, min_count=1, batch_words=4)
    #
    # ############################################################################
    # # Step 8. Create Actor Embedding Features
    # ############################################################################
    # # COMMENTED OUT 2026-02-16: Removing embeddings for interpretability
    # # Create embedding features for 'actor1' and 'actor2' based on the
    # # Node2Vec model. These features capture the latent representations of
    # # actors in the interaction network, enhancing the dataset with
    # # informative attributes for machine learning models.
    # ############################################################################
    #
    # embeddings = {node: model.wv[node] for node in G.nodes()}
    #
    # emb_df = pd.DataFrame.from_dict(embeddings, orient="index")
    # emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
    #
    # # Save embeddings for reproducibility and future inference consistency
    # emb_out = os.path.join(data_path, "actor_embeddings.parquet")
    # emb_df.to_parquet(emb_out)
    # print(f"\nSaved actor embeddings to: {emb_out}")
    #
    # ############################################################################
    # # Step 9. Merge Actor Embeddings with Main DataFrame
    # ############################################################################
    # # COMMENTED OUT 2026-02-16: Removing embeddings for interpretability
    # # Merge the actor embeddings with EACH temporal split to enrich them
    # # with additional features derived from the interaction network.
    # # This integration enhances the dataset, providing more informative
    # # attributes for subsequent machine learning tasks.
    # #
    # # IMPORTANT UPDATE:
    # # We apply embeddings learned from TRAIN graph to VALID/TEST.
    # # Unseen actors will yield NaNs which we convert to zero vectors.
    # ############################################################################
    #
    # train_df = apply_embeddings(train_df, emb_df)
    # valid_df = apply_embeddings(valid_df, emb_df)
    # test_df = apply_embeddings(test_df, emb_df)
    #
    # train_df.fillna(0, inplace=True)
    # valid_df.fillna(0, inplace=True)
    # test_df.fillna(0, inplace=True)
    #
    # #############################################################################
    # # Step 9b. Add Pairwise Embedding Features
    # #############################################################################
    # # COMMENTED OUT 2026-02-16: Removing embeddings for interpretability
    # # Add pairwise embedding features based on the normalized actor names. These features
    # # capture the relationships between pairs of actors involved in events,
    # # providing additional context and information for downstream analysis
    # # and modeling tasks.
    # #############################################################################
    #
    # train_df = add_pairwise_embedding_features(train_df)
    # valid_df = add_pairwise_embedding_features(valid_df)
    # test_df = add_pairwise_embedding_features(test_df)

    ############################################################################
    # Step 10. Re-encode `civilian_targeting` column to binarized format
    ############################################################################
    # The `civilian_targeting` column is re-encoded to a binary format where
    # 1 indicates the presence of civilian targeting and 0 indicates its absence.
    # This transformation simplifies the variable for modeling purposes,
    # making it easier to interpret and utilize in predictive analyses.
    #
    # IMPORTANT UPDATE:
    # Apply transform to EACH split. Use errors-safe approach if column missing.
    ############################################################################

    to_binary = lambda x: 1 if x == "Civilian targeting" else 0
    for _df in (train_df, valid_df, test_df):
        if "civilian_targeting" in _df.columns:
            _df["civilian_targeting"] = _df["civilian_targeting"].apply(to_binary)

    ############################################################################
    # Step 11. Backfill missing admin1 using location
    ############################################################################
    # If admin1 is missing or blank, use location as a proxy regional label.
    # This preserves spatial signal and avoids introducing artificial categories.

    for _df in (train_df, valid_df, test_df):
        if "admin1" in _df.columns and "location" in _df.columns:

            _df["admin1"] = _df["admin1"].replace("", pd.NA).fillna(_df["location"])

            # ensure string dtype for encoder safety
            _df["admin1"] = _df["admin1"].astype(str)

    ############################################################################
    # Step 11b. Days Since Invasion Feature
    ############################################################################
    # Compute the number of days between each event and the start of the
    # full-scale Russian invasion (2022-02-24). This gives the model a
    # monotonic temporal signal without requiring the raw date column.
    ############################################################################

    INVASION_DATE = pd.Timestamp("2022-02-24")

    for _df in (train_df, valid_df, test_df):
        if "event_date" in _df.columns:
            _df["days_since_invasion"] = (
                pd.to_datetime(_df["event_date"]) - INVASION_DATE
            ).dt.days

    # ############################################################################
    # # Step 11c. Distance to Kyiv Feature
    # ############################################################################
    # # Compute the great-circle (haversine) distance in km from each event's
    # # coordinates to Kyiv (50.4501, 30.5234).
    # #
    # # Why haversine over Euclidean:
    # #   Euclidean distance on raw lat/lon is distorted because one degree of
    # #   longitude shrinks with latitude (cos(lat) factor). Ukraine spans
    # #   roughly 44–52°N, so this distortion is non-trivial. Haversine
    # #   accounts for Earth's curvature and returns true surface distance in
    # #   km, giving the model a physically meaningful and unit-consistent
    # #   measure regardless of where in Ukraine the event occurs.
    # #
    # # Why distance to Kyiv matters:
    # #   Kyiv is the political and military command centre of Ukraine.
    # #   Proximity to the capital correlates with strategic importance,
    # #   defensive investment, force concentration, and media/reporting
    # #   density — all of which influence fatality outcomes. While raw
    # #   lat/lon are available, the model would need to learn this radial
    # #   relationship implicitly; providing it as an explicit feature
    # #   reduces the learning burden and improves interpretability, since
    # #   a SHAP value on dist_to_kyiv_km is directly explainable as
    # #   "events closer to / farther from the capital tend to have
    # #   higher/lower predicted fatalities."
    # ############################################################################

    # KYIV_LAT, KYIV_LON = 50.4501, 30.5234

    # for _df in (train_df, valid_df, test_df):
    #     if "latitude" in _df.columns and "longitude" in _df.columns:
    #         _df["dist_to_kyiv_km"] = haversine_km(
    #             _df["latitude"], _df["longitude"], KYIV_LAT, KYIV_LON
    #         )

    ############################################################################
    # Step 12. Drop Intermediate and Redundant Columns
    ############################################################################
    # Drop intermediate columns used for normalization and embedding
    # generation to streamline the dataset and retain only relevant features
    # for modeling. This step helps in reducing redundancy and improving
    # dataset clarity.
    #
    # Also drop event_type because sub_event_type is more granular and informative
    # and `event_date` because it is not needed for regression modeling
    # The full list of columns to be dropped are contained within `drop_vars`
    # inside constants.py
    #
    # IMPORTANT UPDATE:
    # Apply drops to EACH split.
    ############################################################################

    for _df in (train_df, valid_df, test_df):
        _df.drop(columns=drop_vars, errors="ignore", inplace=True)

    ########################################################################
    # Step 13. Ensure Numeric Data and Feature Engineering
    ########################################################################
    # Convert any possible numeric values that may have been incorrectly
    # classified as non-numeric. This avoids accidental labeling errors.
    #
    # IMPORTANT UPDATE:
    # Apply safe_to_numeric to EACH split.
    ########################################################################

    train_df = train_df.apply(lambda x: safe_to_numeric(x))
    valid_df = valid_df.apply(lambda x: safe_to_numeric(x))
    test_df = test_df.apply(lambda x: safe_to_numeric(x))

    ################################################################################
    # Step 14. Zero Variance Columns
    ################################################################################
    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw the following FutureWarning:
    # Dropping of nuisance columns in DataFrame reductions
    # (with 'numeric_only=None') is deprecated; in a future version this will
    # raise TypeError.  Select only valid columns before calling the reduction.
    #
    # IMPORTANT UPDATE:
    # Fit zero-variance columns on TRAIN only, then drop the same columns
    # from VALID and TEST to keep feature space consistent without leakage.
    ################################################################################

    if stage == "training":
        numeric_cols = train_df.select_dtypes(include=["number"]).columns
        var_indf = train_df[numeric_cols].var()
        zero_var = var_indf[var_indf == 0]
        zero_varlist_list = list(zero_var.index)

        dumpObjects(
            zero_varlist_list,
            os.path.join(data_path, "zero_varlist_list.pkl"),
        )

        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
            obj=zero_varlist_list,
        )

    if stage == "inference":
        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    # Drop zero-variance columns from each split (errors ignored)
    train_df = train_df.drop(columns=zero_varlist_list, errors="ignore")
    valid_df = valid_df.drop(columns=zero_varlist_list, errors="ignore")
    test_df = test_df.drop(columns=zero_varlist_list, errors="ignore")

    print(f"Train shape after dropping zero var cols: {train_df.shape}")
    print(f"Valid shape after dropping zero var cols: {valid_df.shape}")
    print(f"Test shape after dropping zero var cols:  {test_df.shape}")

    ############################################################################
    # Step 15. Calculate Row-wise Missingness Percentage
    ############################################################################
    # This step computes the proportion of missing values for each row in the
    # DataFrame. It helps identify rows with a high level of incompleteness, which
    # may be useful for filtering, imputation strategies, or downstream analysis.
    #
    # A new column is added to each split where each value represents
    # the percentage of columns that are missing for that row.
    ############################################################################

    for _df in (train_df, valid_df, test_df):
        _df[percent_miss] = _df.isna().mean(axis=1)

    ############################################################################
    # Step 16 Ensure parquet-safe dtypes
    ############################################################################
    # PyArrow requires consistent column types.
    # Some ACLED admin/location fields may contain mixed types.
    # Convert object columns to string to avoid ArrowTypeError.

    for _df in (train_df, valid_df, test_df):
        obj_cols = _df.select_dtypes(include="object").columns
        _df[obj_cols] = _df[obj_cols].astype(str)

    ############################################################################
    # Step 17. Save Processed Data
    ############################################################################
    ############################################################################

    train_out = os.path.join(data_path, "train.parquet")
    valid_out = os.path.join(data_path, "valid.parquet")
    test_out = os.path.join(data_path, "test.parquet")

    train_df.to_parquet(train_out)
    valid_df.to_parquet(valid_out)
    test_df.to_parquet(test_out)

    print("\nSaved enriched splits (NO EMBEDDINGS - commented out):")
    print(f"  TRAIN: {train_out}")
    print(f"  VALID: {valid_out}")
    print(f"  TEST:  {test_out}")


if __name__ == "__main__":
    app()
