#!/bin/bash

# before, use chmod +x scripts/unziptargz.sh

SOURCE_DIR="./DATASET/Glioma/glioma_224"
DEST_DIR="./DATASET/Glioma/unzip_glioma_224"

# SOURCE_DIR=$(readlink -f "$SOURCE_DIR")
# DEST_DIR=$(readlink -f "$DEST_DIR")

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Navigate to source directory
cd "$SOURCE_DIR"

# Loop through all .tar.gz files in the source directory
for file in *.tar.gz
do
    # echo "Extracted: $file"
    # Ensure the file is an actual file
    if [[ -f "$file" ]]; then
        # Extract file
        # tar -xzvf "$file" -C "$DEST_DIR"
        tar -xzvf "$file"
        echo "Extracted: $file"
    fi
done

echo "Extraction completed!"