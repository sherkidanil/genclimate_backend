"""Simple S3 client using botocore for AWS operations."""

import logging
from typing import BinaryIO

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from config import settings

logger = logging.getLogger(__name__)


class S3Client:
    """S3 client wrapper."""

    def __init__(self):
        """Initialize S3 client with configuration."""
        addresing_style = "virtual" if settings.s3.endpoint_url.startswith("https://obs.") else "path"
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3.endpoint_url,
            region_name=settings.s3.region_name,
            aws_access_key_id=settings.s3.access_key_id,
            aws_secret_access_key=settings.s3.secret_access_key,
            config=Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": addresing_style,
                },
            ),
        )
        self.bucket = settings.s3.bucket_name

    def upload_file(self, file_path: str, object_key: str) -> bool:
        """Upload a file to S3."""

        try:
            with open(file_path, "rb") as file_obj:
                self.client.put_object(Bucket=self.bucket, Key=object_key, Body=file_obj)
            logger.info(f"Successfully uploaded {file_path} to {object_key}")
            return True

        except (BotoCoreError, ClientError, FileNotFoundError) as e:
            logger.error(f"Failed to upload {file_path} to S3: {e}")
            return False

    def upload_fileobj(self, file_obj: BinaryIO, object_key: str) -> bool:
        """Upload a file-like object to S3"""

        try:
            self.client.put_object(Bucket=self.bucket, Key=object_key, Body=file_obj)
            logger.info(f"Successfully uploaded file object to {object_key}")
            return True

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to upload file object to S3: {e}")
            return False

    def get_download_url(self, storage_path: str, expires_in: int = 3 * 60 * 60) -> str | None:
        """Генерирует presigned URL для скачивания файла"""
        try:
            url = self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket, "Key": storage_path},
                ExpiresIn=expires_in,
            )
        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to generate presigned URL for {storage_path}: {e}")
            return None

        logger.info(f"Successfully generated presigned URL for {storage_path}")
        return url


s3_client = S3Client()
