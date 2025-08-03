#!/bin/bash

echo -e "\n📦 Downloading Dataset in Progress..."

DATASET_LINK="https://www.kaggle.com/api/v1/datasets/download/ahmedmohamed365/volleyball"
DEST_DIR="data/volleyball.zip"
EX_DIR="data/volleyball"

# Check if zip file already exists
if [ -f "$DEST_DIR" ]; then
  echo -e "\n⚠️  Dataset already exists at $DEST_DIR"
  echo -n "Do you want to re-download it? (y/n): "
  read answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo -e "\n⏩ Skipping download."
  else
    echo -e "\n🔁 Re-downloading Dataset..."
    curl -L --create-dirs -o "$DEST_DIR" "$DATASET_LINK"
    echo -e "\n✅ Downloading Dataset Completed Successfully"
  fi
else
  curl -L --create-dirs -o "$DEST_DIR" "$DATASET_LINK"
  echo -e "\n✅ Downloading Dataset Completed Successfully"
fi

echo -e "\n📂 Unzipping Files..."

unzip -o "$DEST_DIR" -d "$EX_DIR"

echo -e "\n✅ Done Unzipping Files"