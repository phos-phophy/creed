#!/bin/bash

cuda=

while getopts v:c: flag
do
  case "${flag}" in
    v) cuda=${OPTARG} ;;
    c) config=${OPTARG} ;;
    *) echo "usage: $0 [-v] [-c]" >&2
       exit 1 ;;
  esac
done

run_path="main.py"

export CUDA_VISIBLE_DEVICES=$cuda

python3 $run_path -c "$config"
