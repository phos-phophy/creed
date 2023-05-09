#!/bin/bash

# -------------------Training Shell Script--------------------

# 1. Get parameters
num=4
cuda=

while getopts v:c:s:o: flag
do
  case "${flag}" in
    v) cuda=${OPTARG}
       num=$((num-1));;
    c) config=${OPTARG}
       num=$((num-1));;
    s) seed=${OPTARG}
       num=$((num-1));;
    o) output=${OPTARG}
       num=$((num-1));;
    *) echo "Usage: $0 [-v] [-c] [-s] [-o]" >&2
       exit 1 ;;
  esac
done

# 2. Checks number of parameters
if [[ "$num" != 0 ]]
then
  echo "Not enough parameters" >&2
  echo "Usage: $0 [-v] [-c] [-s] [-o]" >&2
  exit 1
fi

# 3. Runs training
run_path="main.py"
export CUDA_VISIBLE_DEVICES=$cuda
python3 $run_path -c "$config" -s "$seed" -o "$output"
