# ⚽ Football Analytics System

This project performs **football formation detection**, **player & ball tracking**, and computes metrics like **speed**, **distance covered**, and **ball possession** using computer vision and deep learning techniques.

## 📌 Overview

Using match footage as input, this system extracts tactical insights including:
- Player & ball tracking
- Team formation classification
- Camera motion compensation
- Spatial metric analysis (speed, distance)
- Ball possession estimation

It’s built with **YOLOv11**, **OpenCV**, **Ultralytics**, **K-Means**, and **ResNet-18**.


## 🌟 Features

- **YOLOv11-based player and ball detection**
- **Optical flow for stable tracking**
- **ResNet-18 CNN for formation classification**
- **Speed & distance calculations**
- **Possession time tracking**
- **Perspective transformation for spatial analysis**
- **Team segmentation using K-means**


##  How to Run

```bash
git clone https://github.com/aniketGhetla/Football.git

# Run the main script
python main.py --input path/to/video.mp4
