import cv2
import mediapipe as mp
import os
import json
from tqdm import tqdm

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Base directory for input videos and output keypoints
dataset_dir = r'C:\Users\ADMIN\Pictures\Camera Roll\dataset2'  # Folder containing videos like 'asl/hello/1.mp4'
base_keypoints_dir = 'askeypoint2'  # Folder for saving keypoints

# Function to ensure directory structure exists
def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to extract keypoints from landmarks
def extract_keypoints(landmarks):
    keypoints = []
    for landmark in landmarks.landmark:
        keypoints.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        })
    return keypoints

# Process each subdirectory (sign name) in the dataset directory
for sign_name in os.listdir(dataset_dir):
    sign_dir = os.path.join(dataset_dir, sign_name)
    if not os.path.isdir(sign_dir):
        continue  # Skip if not a directory

    # Create the corresponding directory in the output folder (e.g., 'askeypoints/hello')
    sign_output_dir = os.path.join(base_keypoints_dir, sign_name)
    ensure_dir_exists(sign_output_dir)

    # Process video files in the sign directory (supports '.mp4')
    video_files = [f for f in os.listdir(sign_dir) if f.endswith('.mp4')]
    for video_file in tqdm(video_files, desc=f'Processing videos for {sign_name}'):
        video_path = os.path.join(sign_dir, video_file)

        # Open the original video file
        cap = cv2.VideoCapture(video_path)

        # Initialize frame counter for saving each frame's keypoints
        original_keypoints_data = {}  # Collect all frame keypoints for the original video

        # Initialize MediaPipe models
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose, \
             mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:

            frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Process original frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect landmarks for original video
                pose_results = pose.process(rgb_frame)
                hands_results = hands.process(rgb_frame)

                # Collect keypoints for the original frame
                frame_keypoints = {}

                # Extract pose keypoints if detected
                if pose_results.pose_landmarks:
                    pose_keypoints = extract_keypoints(pose_results.pose_landmarks)
                    frame_keypoints['pose'] = pose_keypoints

                # Extract hand keypoints if detected
                if hands_results.multi_hand_landmarks:
                    hand_keypoints = [extract_keypoints(hand_landmarks) for hand_landmarks in hands_results.multi_hand_landmarks]
                    frame_keypoints['hands'] = hand_keypoints

                # Add keypoints for this frame to the original video keypoints data
                original_keypoints_data[f'frame_{frame_counter}'] = frame_keypoints

                frame_counter += 1  # Increment frame counter

        # Release the video capture
        cap.release()

        # Save all the keypoints from the original video to a JSON file
        original_json_filename = f'{os.path.splitext(video_file)[0]}.json'
        original_keypoints_file = os.path.join(sign_output_dir, original_json_filename)
        with open(original_keypoints_file, 'w') as f:
            json.dump(original_keypoints_data, f)

        print(f'Saved original video keypoints: {original_keypoints_file}')
