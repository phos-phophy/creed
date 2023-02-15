#!/bin/bash

cuda=

while getopts v:c:l: flag
do
  case "${flag}" in
    v) cuda=${OPTARG} ;;
    c) config=${OPTARG} ;;
    l) size_limit=${OPTARG} ;;
    *) echo "usage: $0 [-v] [-c] [-l]" >&2
       exit 1 ;;
  esac
done

run_path="./train_large.py"

export CUDA_VISIBLE_DEVICES=$cuda

python3 $run_path -c "$config" -l "$size_limit"
