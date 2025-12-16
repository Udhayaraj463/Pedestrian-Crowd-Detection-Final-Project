# Pedestrian and Crowd Detection – Final Project

## Overview
This repository contains the implementation of a **Pedestrian and Crowd Detection system** developed as a **final project for the Pattern Recognition course**.

The project focuses on detecting pedestrians, tracking their movement, and estimating crowd density in real time using computer vision techniques.

---

## Project Objectives
- Detect pedestrians in video streams
- Track individual pedestrians across frames
- Estimate real-time crowd density in a selected area
- Analyze pedestrian trajectories and movement patterns

---

## Methodology
The system is built using the following methods:

- **Pedestrian Detection:**  
  A YOLO-based deep learning model (`yolo11x.pt`) is used to detect only the person class.

- **Tracking:**  
  Pedestrian tracking is performed using a **Kalman Filter** combined with the **Hungarian matching algorithm**.  
  The tracking configuration is based on `bytetrack.yaml`, ensuring unique ID assignment across frames.

- **Trajectory Analysis:**  
  Pedestrian center points are stored and converted from image coordinates to world coordinates using **homography**.

- **Crowd Density Estimation:**  
  Crowd density is calculated as the number of detected pedestrians divided by the selected measurement area.  
  If no measurement area is defined, a pixel-based density estimation is used as a fallback.

---

## How the System Works
1. Video input is loaded using OpenCV.
2. Each frame is processed for pedestrian detection.
3. Detected pedestrians are tracked across frames using Kalman filtering and Hungarian matching.
4. Trajectories are stored and visualized.
5. Crowd density and pedestrian statistics are computed in real time.

---

## Results
The system produces:
- Real-time pedestrian detection and counting
- Unique pedestrian ID tracking
- Trajectory visualizations and heatmaps
- Crowd density estimation over time
- Speed and temporal analysis plots

---

## Modifications from Base Project
This project is **adapted and extended** from the following open-source repository:

Base Repository:  
https://github.com/pozapas/Crowd-Analyzer

### Key Modifications:
- Extended offline analysis to **real-time crowd monitoring**
- Added live pedestrian counting and unique footfall estimation
- Implemented real-time crowd density calculation
- Integrated trajectory heatmaps and temporal analytics
- Optimized performance for real-time processing
- Preserved scientific post-analysis and automated interpretation

---

## Related Research Paper
This project is related to the following IEEE research paper:

**Crowd Density Estimation using Computer Vision Techniques**  
https://ieeexplore.ieee.org/abstract/document/9963778

The paper discusses pedestrian detection, tracking, and crowd density estimation, which directly aligns with the objectives of this project.

---

## Tools and Libraries
- Python
- OpenCV
- PyTorch
- YOLO
- NumPy
- Pandas
- Matplotlib

---

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/Udhayaraj463/Pedestrian-Crowd-Detection-Final-Project.git
