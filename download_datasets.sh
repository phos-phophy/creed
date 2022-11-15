#!/bin/bash

dataset_path="./etc/datasets"

docred_path=$dataset_path"/docred"

# DocRED
mkdir -p $docred_path"/DocRED_baseline_metadata"
echo "Download DocRED dataset into "$docred_path
wget "https://drive.google.com/uc?export=download&id=1AHUm1-_V9GCtGuDcc8XrMUCJE8B-HHoL" -O $docred_path"/dev.json" -nv
wget "https://drive.google.com/uc?export=download&id=1y9A0zKrvETc1ddUFuFhBg3Xfr7FEL4dW" -O $docred_path"/rel_info.json" -nv
wget "https://drive.google.com/uc?export=download&id=1lAVDcD94Sigx7gR3jTfStI66o86cflum" -O $docred_path"/test.json" -nv
wget "https://drive.google.com/uc?export=download&id=1NN33RzyETbanw4Dg2sRrhckhWpzuBQS9" -O $docred_path"/train_annotated.json" -nv
wget "https://drive.google.com/uc?export=download&id=1Qr4Jct2IJ9BVI86_mCk_Pz0J32ww9dYw" -O $docred_path"/train_distant.json" -nv
wget "https://drive.google.com/uc?export=download&id=1seGEjiIhfAWn9SKPtagfBcEsy7PKoYU5" -O $docred_path"/DocRED_baseline_metadata/word2id.json" -nv
wget "https://drive.google.com/uc?export=download&id=16ATXBfX5N2ngEJpgQAbmMpPm4LW4cp0B" -O $docred_path"/DocRED_baseline_metadata/vec.npy" -nv
wget "https://drive.google.com/uc?export=download&id=11liQ1Mfb8hpy5rOobtDDSF_rxFrld1Os" -O $docred_path"/DocRED_baseline_metadata/rel2id.json" -nv
wget "https://drive.google.com/uc?export=download&id=1bA3RsQWXkAzlVETwoVR9YqufamilTSxa" -O $docred_path"/DocRED_baseline_metadata/ner2id.json" -nv
wget "https://drive.google.com/uc?export=download&id=1vET-iiIGQj5mmkQYRH-6-9yVozOFjdN3" -O $docred_path"/DocRED_baseline_metadata/char2id.json" -nv
wget "https://drive.google.com/uc?export=download&id=1Hs2ypMENAyfnL0tzESlHJg1ZbVFfmPL8" -O $docred_path"/DocRED_baseline_metadata/char_vec.npy" -nv
