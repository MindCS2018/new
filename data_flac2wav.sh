#!/bin/bash

folder=../database/LibriSpeech/LibriSpeech_train_clean_100

for file in $(find "$folder" -type f -iname "*.flac")
do
    name=$(basename "$file" .flac)
    dir=$(dirname "$file")
    echo ffmpeg -i "$file" "$dir"/"$name".wav
    ffmpeg -i $file $dir/$name.wav
    #rm $file
done
