"""
Save the model to GCS.
"""

from pathlib import Path

from loguru import logger
from result import Ok, Result
import google.cloud.storage as storage


def save_dir(
    dir_path: Path,
    gcs_path: str,
    bucket_name: str,
) -> Result[None, str]:
    """Save the directory to GCS."""
    bucket = storage.Client().bucket(bucket_name=bucket_name)
    logger.info(f"Saving {dir_path} to GCS to {gcs_path}")
    for file_path in Path(dir_path).glob("*"):
        logger.info(f"Saving {file_path} to GCS")
        blob = bucket.blob(blob_name=f"{gcs_path}/{file_path.name}")
        blob.upload_from_filename(filename=file_path)
    return Ok(None)
