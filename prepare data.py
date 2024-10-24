import numpy as np
import json
import os

# Function to load and preprocess keypoint data
def load_keypoint_data(directory):
    sequences, labels = [], []
    for sign_name in os.listdir(directory):
        sign_dir = os.path.join(directory, sign_name)
        for file in os.listdir(sign_dir):
            if file.endswith('.json'):
                filepath = os.path.join(sign_dir, file)
                with open(filepath, 'r') as f:
                    keypoints = json.load(f)
                    
                # Extract frames and keypoints for each frame
                frames = []
                for frame in keypoints.values():
                    pose_keypoints = frame.get('pose', [])
                    if pose_keypoints:  # Only include if pose keypoints are available
                        frame_keypoints = [[kp['x'], kp['y'], kp['z']] for kp in pose_keypoints]
                        frames.append(frame_keypoints)

                # Add frames only if any keypoints were detected
                if frames:
                    sequences.append(frames)
                    labels.append(sign_name)
    
    return sequences, np.array(labels)

# Load the data
data_dir = 'askeypoints'
sequences, labels = load_keypoint_data(data_dir)

# Print shapes for verification
print(f'Total sequences: {len(sequences)}')
print(f'Total labels: {len(labels)}')
