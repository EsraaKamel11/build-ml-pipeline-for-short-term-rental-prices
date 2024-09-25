#!/usr/bin/env python
"""
This script downloads a file from a URL to a local destination and logs the artifact
to Weights & Biases.

Usage:
    python run.py <sample> <artifact_name> <artifact_type> <artifact_description>

Arguments:
    --sample (str): Name of the sample to download.
    --artifact_name (str): Name for the output artifact.
    --artifact_type (str): Type of the output artifact.
    --artifact_description (str): A brief description of this artifact.
"""

import argparse
import logging
import os
import wandb
from wandb_utils.log_artifact import log_artifact

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Downloads the specified sample file and uploads it to Weights & Biases.

    Args:
        args (Namespace): Command-line arguments parsed from the input.

    This function initializes a W&B run, logs the provided sample file as an artifact,
    and logs information about the operation.

    Raises:
        Exception: If there are issues with the artifact upload.
    """
    run = wandb.init(project="nyc_airbnb", job_type="download_file")
    run.config.update(args)

    logger.info(f"Returning sample: {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")

    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        os.path.join("data", args.sample),
        run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("sample", type=str, help="Name of the sample to download"
    )
    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("artifact_type", type=str, help="Output artifact type.")
    parser.add_argument("artifact_description", type=str, help="A brief description of this artifact")

    args = parser.parse_args()

    go(args)
