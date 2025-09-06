#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p pretrained_weights
mkdir -p data_splits
mkdir -p results

# Download pretrained model (using a more reliable source)
cd pretrained_weights
if [ ! -f "COVER.pth" ]; then
    echo "Downloading COVER pretrained model..."
    # Try multiple sources for the model
    wget -O COVER.pth "https://huggingface.co/vztu/COVER/resolve/main/COVER.pth" || \
    wget -O COVER.pth "https://github.com/vztu/COVER/releases/download/v1.0/COVER.pth" || \
    echo "Warning: Could not download pretrained model. Please download manually."
fi
cd ..

# Create test video IDs file if it doesn't exist
if [ ! -f "data_splits/test_video_ids.txt" ]; then
    echo "Creating test video IDs file..."
    # Extract unique flickr_ids from scores_duplicated.csv for testing
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
