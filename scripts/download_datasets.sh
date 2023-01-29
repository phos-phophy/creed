#!/bin/bash

dataset_path="./etc/datasets"

docred_path=$dataset_path"/docred"

# DocRED: https://github.com/thunlp/DocRED
echo "Download DocRED dataset into "$docred_path
gdown --folder https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw -O $docred_path
