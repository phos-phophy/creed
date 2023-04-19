#!/bin/bash

cuda=

while getopts v:c:s:o: flag
do
  case "${flag}" in
    v) cuda=${OPTARG} ;;
    c) config=${OPTARG} ;;
    s) seed=${OPTARG} ;;
    o) output=${OPTARG} ;;
    *) echo "usage: $0 [-v] [-c] [-s] [-o]" >&2
       exit 1 ;;
  esac
done

run_path="main.py"

export CUDA_VISIBLE_DEVICES=$cuda

python3 $run_path -c "$config" -s "$seed" -o "$output"
