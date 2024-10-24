import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping


max_frames=60
# Load the keypoints and labels
def load_data(keypoints_dir):
    data = []
    labels = []
    
    # Define the number of keypoints per hand and pose
    num_hand_keypoints = 21 * 2  # 21 landmarks for each hand (x, y pairs), 2 hands
    num_pose_keypoints = 33 * 3   # 33 landmarks for pose (x, y, z) 

    max_frames = 60  # Limit to 2.5 seconds at 24 fps

    for sign_name in os.listdir(keypoints_dir):
        sign_dir = os.path.join(keypoints_dir, sign_name)
        
        if not os.path.isdir(sign_dir):
            continue  # Skip if not a directory

        # Read all JSON files for each sign
        for json_file in os.listdir(sign_dir):
            if json_file.endswith('.json'):
                json_path = os.path.join(sign_dir, json_file)
                with open(json_path, 'r') as f:
                    keypoints = json.load(f)
                    # Prepare a fixed-length array for each frame
                    frames = []
                    for frame in keypoints.values():
                        frame_data = np.zeros(num_hand_keypoints + num_pose_keypoints)  # Initialize with zeros
                        
                        # Extract hand keypoints
                        if 'hands' in frame:
                            for i, hand in enumerate(frame['hands']):
                                if i < 2:  # Ensure we only process two hands
                                    hand_keypoints = [landmark['x'] for landmark in hand] + [landmark['y'] for landmark in hand]
                                    frame_data[i * 42:i * 42 + len(hand_keypoints)] = hand_keypoints  # Fill in hand keypoints

                        # Extract pose keypoints
                        if 'pose' in frame:
                            pose_keypoints = []
                            for landmark in frame['pose']:
                                pose_keypoints.extend([landmark['x'], landmark['y'], landmark['z']])
                            if len(pose_keypoints) == num_pose_keypoints:  # Ensure correct length
                                frame_data[num_hand_keypoints:] = pose_keypoints  # Fill in pose keypoints
                            else:
                                # Handle cases where pose keypoints are missing
                                print(f"Warning: Expected {num_pose_keypoints} pose keypoints, got {len(pose_keypoints)} instead.")
                                frame_data[num_hand_keypoints:] = np.zeros(num_pose_keypoints)  # Fill with zeros if not enough keypoints

                        frames.append(frame_data)  # Add to frames list

                    # Limit to max_frames
                    if len(frames) > max_frames:
                        frames = frames[:max_frames]
                    elif len(frames) < max_frames:
                        # If there are fewer frames, pad with zeros
                        frames += [np.zeros(num_hand_keypoints + num_pose_keypoints)] * (max_frames - len(frames))

                    # Store the frames and corresponding label
                    data.append(frames)
                    labels.append(sign_name)

    return data, labels

# Load keypoints data and labels
keypoints_dir = 'datasets/askeypoint2'  # Folder containing the extracted keypoints
data, labels = load_data(keypoints_dir)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert to numpy arrays
X = np.array(data)
y = np.array(encoded_labels)

# No need to pad sequences since we already fixed the length
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_frames, X.shape[2])),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Set up Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the model
model.save('asl_model.h5')
model.summary

