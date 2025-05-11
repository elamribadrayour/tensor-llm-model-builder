"""
Save functions for the builder.
"""

from pathlib import Path

from loguru import logger
from result import Ok, Result
import google.cloud.storage as storage


def save_dir(bucket_name: str, gcs_path: str, dir_path: Path) -> Result[None, str]:
    """Save a directory to google cloud storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name=bucket_name)

    logger.info(f"Saving {dir_path} to GCS to {gcs_path}")
    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():  # Only upload files, not directories
            continue
        relative_path = file_path.relative_to(dir_path)
        gcs_file_path = f"{gcs_path}/{relative_path}"
        logger.info(f"Saving {file_path} to GCS as {gcs_file_path}")
        blob = bucket.blob(blob_name=gcs_file_path)
        blob.upload_from_filename(filename=file_path)

    return Ok(None)
