#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import wandb
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="nyc_airbnb", job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).download()

    df = pd.read_csv(os.path.join(artifact_local_path, "clean_sample.csv"))

    logger.info(f"Performing train-test split with test size = {args.test_size} and stratification on '{args.stratify_by}'.")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save the datasets in the artifact_local_path
    for df, k in zip([trainval, test], ['trainval', 'test']):
        output_file = os.path.join(artifact_local_path, f"{k}_data.csv")
        logger.info(f"Uploading {k}_data.csv dataset to W&B.")
        df.to_csv(output_file, index=False)

        log_artifact(
                f"{k}_data.csv",
                f"{k}_data",
                f"{k} split of dataset",
                output_file,
                run,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
