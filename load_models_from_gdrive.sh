#!/bin/bash

download() {
    file_path=$1
    gdrive_file_id=$2

    if [ ! -f "$file_path" ]
    then
        gdown "$gdrive_file_id" -O "$file_path"
    fi

    file_name=$(basename "$file_path")
    echo "$file_name" "was downloaded"
}

download "models/lightfm/light_fm.dill" "1NjTwM9hMveiV8twsfmcElswkj6iAZnsx"

download "models/lightfm/ann/user_embeddings.dill" "107WdHB8Ka6Hqupw4g3iALLEG2w3N52y9"

download "models/popular/popular_in_category.dill" "1roodlsVXvouTbAbtx57SRBIMx38t8GRO"

download "models/knn/user_knn.dill" "1gDrHPzSYB12CXXKKaZRxlHQW6bcedv48"
