# Uploading Scripts to S3

Training scripts are stored in the ML data S3 bucket and mounted into pods via the S3 CSI driver.

## Upload

# Upload a single script

aws s3 cp scripts/PATH_TO_SCRIPT.sh s3://ethrc-ml-scripts/PATH_TO_SCRIPT.sh

# Upload an entire folder

aws s3 cp scripts/FOLDER_NAME s3://ethrc-ml-scripts/FOLDER_NAME --recursive
