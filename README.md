# Emotion-Based Image Transition with OpenCV and TensorFlow

This project uses a webcam to detect facial emotions in real-time and smoothly transitions between corresponding images for detected emotions. It utilizes a pre-trained TensorFlow model (`emotion_model.hdf5`) for emotion detection and OpenCV for video processing and visualization.

## Features
- Real-time emotion detection using a webcam.
- Smooth transitions between images representing different emotions.
- Supports the following emotions:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

## Requirements
- Python 3.10 or higher
- A functional webcam
- Pre-trained emotion model (`emotion_model.hdf5`)
- Corresponding images for each emotion (`angry.jpg`, `disgust.jpg`, etc.)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/emotion-based-art.git
   python emotion.py
