#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in Weights & Biases
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np
import os
from wandb_utils.log_artifact import log_artifact


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact.
    logger.info(f'Downloading input artifact {args.input_artifact}')
    artifact_local_path = run.use_artifact(args.input_artifact).download()
    df_sample = pd.read_csv(os.path.join(artifact_local_path, args.sample))

    # drop outliers
    logger.info(f'Dropping outliers regarding min {args.min_price}, max {args.max_price} price thresholds')
    min_price = args.min_price
    max_price = args.max_price
    idx = df_sample['price'].between(min_price, max_price)
    df_clean = df_sample[idx].copy()

    # normal distribution of 'minimum_nights'
    logger.info("Transforming skewness of feature 'minimum_nights' to normal distribution")
    df_clean['minimum_nights'] = np.log(df_clean['minimum_nights'])

    # convert 'last_review' to datetime
    logger.info('Converting feature "last_review" to datetime type')
    df_clean['last_review'] = pd.to_datetime(df_clean['last_review'])

    # save cleaned dataframe
    logger.info(f'Saving cleaned dataframe as {args.output_artifact}')
    df_clean.to_csv(os.path.join(artifact_local_path,args.output_artifact),index=False)

    # log artifact to Weights & Biases
    logger.info(f'Logging artifact {args.output_artifact} to W&B.')

    log_artifact(
        args.output_artifact,
        args.output_artifact_type,
        args.output_artifact_description,
        os.path.join(os.path.join(artifact_local_path,args.output_artifact)),
        run,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data.")

    parser.add_argument(
        "--sample",
        type=str,
        help="Name of the sample dataset"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The input artifact to clean",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The name of the cleaned output artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_type",
        type=str,
        help="The type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_description",
        type=str,
        help="A description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to consider for cleaning",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to consider for cleaning",
        required=True
    )

    args = parser.parse_args()

    go(args)
