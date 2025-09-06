#!/bin/bash

# Install dependencies
pip install -r requirements.txt -q

# Create necessary directories
mkdir -p pretrained_weights
mkdir -p data_splits
mkdir -p results

# Download pretrained model using the working raw URL
cd pretrained_weights
if [ ! -f "COVER.pth" ] || [ ! -s "COVER.pth" ]; then
    echo "Downloading COVER pretrained model..."
    rm -f COVER.pth  # Remove corrupted file if exists
    
    # Use the working raw GitHub URL
    wget -O COVER.pth "https://github.com/vztu/COVER/raw/release/Model/COVER.pth"
    
    # Check if download was successful
    if [ -f "COVER.pth" ] && [ -s "COVER.pth" ]; then
        echo "Model downloaded successfully. Size: $(du -h COVER.pth | cut -f1)"
    else
        echo "ERROR: Failed to download pretrained model!"
        exit 1
    fi
else
    echo "Model already exists. Size: $(du -h COVER.pth | cut -f1)"
fi
cd ..

# Create test video IDs file if it doesn't exist
if [ ! -f "data_splits/test_video_ids.txt" ]; then
    echo "Creating test video IDs file..."
    python -c "
import pandas as pd
import numpy as np

# Read the scores file
df = pd.read_csv('scores/scores_duplicated.csv')
# Get unique flickr_ids and take first 100 for testing
test_ids = df['flickr_id'].unique()[:100]
# Save to file
with open('data_splits/test_video_ids.txt', 'w') as f:
    for vid_id in test_ids:
        f.write(f'{vid_id}\n')
print(f'Created test_video_ids.txt with {len(test_ids)} video IDs')
"
fi

echo "Setup completed successfully!"
echo "You can now run: python final_evaluation.py"
