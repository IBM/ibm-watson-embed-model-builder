#!/usr/bin/env bash
# A lot of this script comes from StackOverflow
# CITE: https://stackoverflow.com/a/44751929

# Pull the input object and determine if unzipping
input_path=${1:-"/app/model.zip"}
model_root_dir=${MODEL_ROOT_DIR:-"/app/models"}
upload=${UPLOAD:-"false"}
insecure_mode=${USE_INSECURE_UPLOADS:-"false"}

insecure_arg=""
if [ $insecure_mode = "true" ]
then
  insecure_arg="-k"
fi

# From here out, fail on unresolved variables and broken pipes!
set -euo pipefail

# The target name for the model when unpacked
model_name=$MODEL_NAME

## Helpers #####################################################################

function upload {
    upload_file=$1
    upload_path=$UPLOAD_PATH

    # Curl call args
    resource="/$bucket"
    if [ "$upload_path" != "" ]
    then
        resource="$resource/$upload_path"
    fi
    resource="$resource/$(basename $upload_file)"
    content_type="application/octet-stream"
    date_value=$(date -R)
    signing_string="PUT\n\n$content_type\n$date_value\n$resource"
    signature=$(echo -en $signing_string | openssl sha1 -hmac $secret -binary | base64)

    # Curl call
    echo "Uploading [$upload_file -> $resource]"
    curl -X PUT -T "$upload_file" \
        ${insecure_arg} \
        -H "Host: $(echo $url | rev | cut -d'/' -f 1 | rev)" \
        -H "Date: $date_value" \
        -H "Content-Type: $content_type" \
        -H "Authorization: AWS $key:$signature" \
        ${url}${resource}
}

## Main ########################################################################

# Make the target output directory in the model root
model_dir="${model_root_dir}/${model_name}"
mkdir -p "$model_dir"
cd $model_dir

# Unzip the target file
unzip -o $input_path

# Update the input path to trigger the upload
cd ..
input_path=$(echo $model_dir | sed "s,^$model_root_dir,," | sed "s,^/,,")

# If uploading, do the upload
if [ "$upload" == "true" ]
then
    storage_config_file=${S3_CONFIG_FILE:-""}
    if [ -f $storage_config_file ]
    then
        # Using -r (raw) to not include the quotes around the values
        url=$(jq -r .endpoint_url "$storage_config_file")
        key=$(jq -r .access_key_id "$storage_config_file")
        secret=$(jq -r .secret_access_key "$storage_config_file")
        bucket=$(jq -r .default_bucket "$storage_config_file")
    else
        url=$S3_URL
        key=$S3_ACCESS_KEY_ID
        secret=$S3_SECRET_ACCESS_KEY
        bucket=$S3_BUCKET_NAME
    fi

    base_upload_path=$UPLOAD_PATH

    # If the upload path is a single file, just upload it
    if [ -f $input_path ]
    then
        upload $input_path

    # If it's a directory, upload all files within in
    elif [ -d $input_path ]
    then
        for fname in $(find $input_path)
        do
            if ! [ -d $fname ]
            then
                upload_path=""
                if [ "$base_upload_path" != "" ]
                then
                    upload_path="$base_upload_path/"
                fi
                upload_path="$upload_path$(dirname $fname)"
                UPLOAD_PATH=$upload_path upload $fname
            fi
        done
    fi
fi
