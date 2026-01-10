################################################################################
######################### Import Requisite Libraries ###########################
import os
import typer
import re
import pandas as pd
from node2vec import Node2Vec

# import pickling scripts
from model_tuner.pickleObjects import dumpObjects

################################################################################

from core.constants import (
    var_index,
    preproc_run_name,
    exp_artifact_name,
    percent_miss,
    seed,
    drop_vars,
)

# import all user-defined functions and constants
from core.functions import (
    mlflow_dumpArtifact,
    mlflow_loadArtifact,
    safe_to_numeric,
    normalize_actor,
    build_actor_interaction_graph,
    add_pairwise_embedding_features,
)

app = typer.Typer()


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

    df = pd.read_parquet(input_data_file)

    try:
        df.set_index(var_index, inplace=True)
    except:
        print("Index already set or 'var_index' doesn't exist in dataframe")

    print(f"Input Data Shape: {df.shape}")
    print(f"There are {df.index.unique().shape[0]} unique indices in the dataframe.")

    if stage == "training":

        df_object = df.select_dtypes("object")
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

        ########################################################################
        # Step 2. String Columns Handling
        ########################################################################
        # String columns are identified and should be removed before modeling
        # because machine learning models typically require numerical inputs.
        # Keeping string columns in the dataset may lead to errors or
        # unintended behavior unless explicitly encoded.
        #
        # To ensure consistency between training and inference,
        # we save the list of string columns and track it using MLflow.
        ########################################################################

        # Extract column names to a list
        string_cols_list = df_object.columns.to_list()

        ########################################################################
        # Step 3. Save and Log String Column List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        # This list of string columns is dumped (stored) only to inform of what
        # the string columns are; no further action is taken; we do not need to
        # load this list into production, since it is only there for us to
        # see what the columns are.
        ########################################################################

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
    ############################################################################

        actor1_list = df["actor1"].unique().tolist()
        actor2_list = df["actor2"].unique().tolist()

        # Dump the actor1_list into a pickle file for future reference
        dumpObjects(
                actor1_list,
                os.path.join(data_path, "actor1_list.pkl"),
            )
    
            # Dump the actor1_list into a pickle file for future reference
        dumpObjects(
                actor2_list,
                os.path.join(data_path, "actor2_list.pkl"),
            )
        
        # Log the actor1_list as an artifact in MLflow
        mlflow_dumpArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="actor1_list",
            obj=actor1_list,
        )

        # Log the actor2_list as an artifact in MLflow
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
        # During training, we identified and stored marital_status.
        # Now, we reload this to ensure that inference follows the same
        # preprocessing pipeline as training, maintaining consistency.
        ########################################################################

        ## Load marital_status from artifacts
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

    df["actor1_root"] = df["actor1"].apply(normalize_actor)
    df["actor2_root"] = df["actor2"].apply(normalize_actor)

    # We are keeping actor_2_missing indicator for possible future use b/c 
    # it answers the question of whether an event involved a known actor2 or not.
    # Is there an explicit second actor in this event?

    df["actor2_missing"] = df["actor2"].isna().astype(int)

    print(f"\nTop 10 Normalized Actor 1 Names: \n{df['actor1_root'].value_counts().head(10)}")
    print(f"\nTop 10 Normalized Actor 2 Names: \n{df['actor2_root'].value_counts().head(10)}")

    ############################################################################
    # Step 6. Build Actor Interaction Graph
    ############################################################################
    # Construct a graph representing interactions between actors based on
    # co-occurrences in events. This graph captures relationships and patterns
    # among actors, which can be useful for network analysis and feature
    # engineering in predictive modeling.

    G = build_actor_interaction_graph(df)

    ############################################################################
    # Step 7. Generate Node2Vec Embeddings
    ############################################################################
    # Generate Node2Vec embeddings for actors in the interaction graph.
    # These embeddings capture the structural relationships and properties
    # of actors within the network, providing valuable features for downstream
    # machine learning tasks.
    ############################################################################

    node2vec = Node2Vec(
        G,
        dimensions=32,
        walk_length=10,
        num_walks=50,
        workers=1,
        weight_key="weight",
        seed=seed,
    )

    model = node2vec.fit(
        window=5,
        min_count=1,
        batch_words=4
    )

    ############################################################################
    # Step 8. Create Actor Embedding Features
    ############################################################################
    # Create embedding features for 'actor1' and 'actor2' based on the
    # Node2Vec model. These features capture the latent representations of
    # actors in the interaction network, enhancing the dataset with
    # informative attributes for machine learning models.   
    ############################################################################
    embeddings = {
    node: model.wv[node]
    for node in G.nodes()
    }

    emb_df = pd.DataFrame.from_dict(embeddings, orient="index")
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

    ############################################################################
    # Step 9. Merge Actor Embeddings with Main DataFrame
    ############################################################################
    # Merge the actor embeddings with the main DataFrame to enrich it
    # with additional features derived from the interaction network.
    # This integration enhances the dataset, providing more informative
    # attributes for subsequent machine learning tasks.

    df = df.merge(
        emb_df,
        left_on="actor1_root",
        right_index=True,
        how="left"
    )

    # Actor 2 embeddings with prefix 'a2_' because emb_ is already used for actor1
    df = df.merge(
        emb_df.add_prefix("a2_"),
        left_on="actor2_root",
        right_index=True,
        how="left"
    )

    #############################################################################
    # Step 9b. Add Pairwise Embedding Features
    #############################################################################
    # Add pairwise embedding features based on the normalized actor names. These features
    # capture the relationships between pairs of actors involved in events,
    # providing additional context and information for downstream analysis
    # and modeling tasks.
    #############################################################################

    df = add_pairwise_embedding_features(df=df)

    ############################################################################
    # Step 10. Drop Intermediate and Redundant Columns
    ############################################################################
    # Drop intermediate columns used for normalization and embedding
    # generation to streamline the dataset and retain only relevant features
    # for modeling. This step helps in reducing redundancy and improving
    # dataset clarity.

    # Also drop event_type because sub_event_type is more granular and informative
    # and `event_date` because it is not needed for regression modeling
    # The full list of columns to be dropped are contained within `drop_vars` 
    # inside constants.py

    df.drop(columns=drop_vars, inplace=True)

    ############################################################################
    # Step 11. Re-encode `civilian_targeting` column to binarized format
    ############################################################################
    # The `civilian_targeting` column is re-encoded to a binary format where
    # 1 indicates the presence of civilian targeting and 0 indicates its absence.
    # This transformation simplifies the variable for modeling purposes,
    # making it easier to interpret and utilize in predictive analyses.

    to_binary = lambda x: 1 if x == "Civilian targeting" else 0
    df["civilian_targeting"] = df["civilian_targeting"].apply(to_binary)

    ########################################################################
    # Step 12. Ensure Numeric Data and Feature Engineering
    ########################################################################
    # Convert any possible numeric values that may have been incorrectly
    # classified as non-numeric. This avoids accidental labeling errors.
    ########################################################################

    # Convert possible numeric columns to actual numeric types
    df = df.apply(lambda x: safe_to_numeric(x))


    ############################################################################
    # Step 13. Concatenate Tags Column to Notes Column
    ############################################################################
    # Combine the 'tags' column with the 'notes' column to consolidate
    # relevant information into a single text field. This concatenation
    # enhances the dataset by merging related textual data, which can be
    # beneficial for natural language processing tasks or feature extraction. 
    # Retaining tags by itself may lead to sparse data issues.

    df["tags"] = df["tags"].fillna("").astype(str)

    df["notes"] = (
        df["notes"].fillna("").astype(str)
        + " | TAGS: "
        + df["tags"]
    )
    df.drop(columns=["tags"], inplace=True) ## drop tags after concatenation

    ################################################################################
    # Step 14. Zero Variance Columns
    ################################################################################

    # Select only numeric columns s/t .var() can be applied since you can only
    # call this function on numeric columns; otherwise, if you include a mix
    # (object and numeric), it will throw the following FutureWarning:
    # Dropping of nuisance columns in DataFrame reductions
    # (with 'numeric_only=None') is deprecated; in a future version this will
    # raise TypeError.  Select only valid columns before calling the reduction.

    ################################################################################

    if stage == "training":
        # Extract numeric columns to compute variance and identify
        # zero-variance features
        numeric_cols = df.select_dtypes(include=["number"]).columns
        var_indf = df[numeric_cols].var()

        # identify zero variance columns
        zero_var = var_indf[var_indf == 0]

        # capture zero-variance cols in list
        zero_varlist_list = list(zero_var.index)

        ########################################################################
        # Step 8. Save and Log Zero Variance Columns List
        ########################################################################
        # Save the list of string columns for consistency across training and
        # inference and log them in MLflow for reproducibility.
        ########################################################################

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

        ########################################################################
        # Load Previously Saved Zero Variance Columns List
        ########################################################################

        # load zero_var_list
        zero_varlist_list = mlflow_loadArtifact(
            experiment_name=exp_artifact_name,
            run_name=preproc_run_name,
            obj_name="zero_varlist_list",
        )

    ########################################################################
    # Step 15. Remove zero variance cols from main df, and assign to new var
    # df_sans_zero
    ########################################################################
    df_sans_zero = df.drop(columns=zero_varlist_list)

    print(f"Sans Zero Var Shape: {df_sans_zero.shape}")

    print()
    print(f"Original shape: {df.shape[1]} columns.")
    print(f"Reduced by {df.shape[1]-df_sans_zero.shape[1]} zero variance columns.")
    print(f"Zero Variance Columns: {zero_varlist_list}")
    print(f"Now there are {df_sans_zero.shape[1]} columns.")
    print()


    df_sans_zero_missing = df_sans_zero.copy()  

    print(f"Sans Zero Data Shape: {df_sans_zero_missing.shape}")



    ############################################################################
    # Step 16. Calculate Row-wise Missingness Percentage
    ############################################################################
    # This step computes the proportion of missing values for each row in the
    # DataFrame. It helps identify rows with a high level of incompleteness, which
    # may be useful for filtering, imputation strategies, or downstream analysis.
    #
    # A new column is added to `df_sans_zero_missing` where each value represents
    # the percentage of columns that are missing for that row.
    ############################################################################

    df_sans_zero_missing[percent_miss] = df_sans_zero_missing.isna().mean(axis=1)

    ############################################################################
    # Step 17. Save Processed Data
    ############################################################################

    # Save out the dataframe to parquet file
    print(f"df_sans_zero_missing.shape after adding {percent_miss}: "
          f"{df_sans_zero_missing.shape}")
    df_sans_zero_missing.reset_index().to_parquet(output_data_file)


if __name__ == "__main__":
    app()