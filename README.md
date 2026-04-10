# Automated Poker Dice Recognition and Robotic Sorting Using Convolutional Neural Networks and Classical Computer Vision

> **Course:** Computer Vision Laboratory  
> **Program:** Intelligent Systems Engineering, 3rd Year  
> **Institution:** Intercontinental University of Enterprise (uie.edu)


> 
---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#1-introduction)
3. [System Architecture](#2-system-architecture)
4. [Methodology](#3-methodology)
   - 3.1 [Workspace Calibration](#31-workspace-calibration)
   - 3.2 [Dice Segmentation](#32-dice-segmentation)
   - 3.3 [Dataset Acquisition](#33-dataset-acquisition)
   - 3.4 [CNN Architecture and Training](#34-cnn-architecture-and-training)
   - 3.5 [Poker Hand Evaluation](#35-poker-hand-evaluation)
   - 3.6 [Robotic Pick-and-Place](#36-robotic-pick-and-place)
5. [Results](#4-results)
6. [Discussion](#5-discussion)
7. [Project Structure](#6-project-structure)
8. [Installation and Usage](#7-installation-and-usage)
9. [Conclusions](#8-conclusions)
10. [References](#9-references)

---

## Abstract

This work presents an end-to-end robotic vision system that integrates classical computer vision techniques with deep learning for the autonomous recognition and physical sorting of poker dice. The system operates on a Niryo One robotic arm equipped with a monocular camera and performs four sequential tasks: (i) workspace calibration via circular fiducial marker detection and homography estimation, (ii) dice segmentation through edge-based contour analysis, (iii) face classification using a lightweight convolutional neural network (CNN) exported to ONNX format, and (iv) robotic pick-and-place operations that physically group dice according to the evaluated poker hand. The CNN, trained on a custom dataset of 1,007 labelled images across six classes, achieves a validation accuracy of 98.5%. The complete pipeline operates in real time on the robot's control workstation, demonstrating the feasibility of combining geometric calibration, learned perception, and robotic manipulation within a unified framework.

---

## 1. Introduction

Robotic manipulation guided by visual perception remains a central challenge in intelligent systems, requiring the integration of multiple disciplines: image processing, machine learning, and motion planning. In the context of this laboratory, the objective extends beyond shape detection (addressed in a preceding lab) to incorporate **semantic scene understanding** -- the ability to not only locate objects but to interpret their meaning and act upon that interpretation.

The specific task is formulated as follows: given a set of five poker dice placed arbitrarily on a calibrated workspace, the system must (1) detect and localise each die, (2) classify the face shown on each die from the set {9, 10, J, Q, K, A}, (3) evaluate the resulting poker hand according to standard poker dice rules, and (4) autonomously pick and place each die into a column-based arrangement that visually communicates the hand structure.

This report documents the design, implementation, and evaluation of the complete pipeline, with emphasis on the interplay between classical computer vision methods and a trained neural classifier.

---

## 2. System Architecture

The system follows a modular pipeline architecture in which each stage transforms the data representation from raw pixel input to physical robot actions. The design enforces a strict separation of concerns through a **normalised coordinate space** $[0, 1]^2$ that decouples the vision subsystem from the robot controller.

![System architecture](assets/media/niryo_app_architecture.svg)
*Figure 1. High-level system architecture. The camera captures a frame, the vision module extracts the workspace and segments the dice, the CNN classifier assigns face labels, the evaluator determines the poker hand, and the robot controller executes pick-and-place operations.*

The architecture comprises five principal modules:

| Module | File | Responsibility |
|--------|------|----------------|
| **Vision** | `src/vision.py` | Workspace detection, homography computation, dice segmentation |
| **Classifier** | `src/classifier.py` | ONNX-based CNN inference for face recognition |
| **Evaluator** | `src/evaluator.py` | Poker hand determination from classified faces |
| **Robot Controller** | `src/robot.py` | Coordinate mapping and robotic arm control via `pyniryo` |
| **Orchestrator** | `src/poker.py` | Main event loop, HUD rendering, and command processing |

Supporting modules include `src/train.py` (CNN training pipeline), `src/capture.py` (dataset acquisition tool), `src/config.py` (centralised parameters), and `src/dataset.py` (data utilities).

---

## 3. Methodology

### 3.1 Workspace Calibration

The workspace is defined by four concentric circular fiducial markers (dianas) printed at the corners of a flat board. Detection proceeds as follows:

1. **Fiducial detection.** The Hough Circle Transform is applied to the grayscale frame to detect circular targets.
2. **Corner assignment.** Detected circles are sorted by proximity to the image corners to establish a consistent labelling: top-left (TL), top-right (TR), bottom-right (BR), bottom-left (BL).
3. **Homography estimation.** A perspective transformation matrix $H$ is computed that maps pixel coordinates $(u, v)$ to normalised workspace coordinates $(x_n, y_n) \in [0, 1]^2$.

This homography serves as the central abstraction layer: all downstream modules operate exclusively in the normalised space, enabling independent development and testing of the vision and robot subsystems. The inverse mapping from normalised coordinates to robot-frame coordinates $(x_r, y_r)$ is performed via bilinear interpolation over four calibrated corner poses.

### 3.2 Dice Segmentation

Dice segmentation presents a non-trivial challenge in this setup, as both the dice and the workspace surface are white, precluding colour-based segmentation. An edge-based approach is employed instead:

1. **Edge detection.** A Canny edge detector is applied to the grayscale image.
2. **Region masking.** Edges outside the workspace polygon are suppressed.
3. **Morphological refinement.** Dilation and morphological closing operations consolidate edge fragments into closed contours.
4. **Contour filtering.** Contours are filtered by area (1,000 -- 150,000 px$^2$) and by distance from fiducial markers (exclusion radius of 50 px) to eliminate false positives.
5. **Crop extraction.** For each valid contour, the bounding rectangle is extracted with 15 px padding and passed to the classifier.

![Dice identification -- static frame](assets/media/1_1identificacion.png)
*Figure 2. Dice detection on a static frame. Each segmented die is bounded and labelled with the CNN prediction.*

### 3.3 Dataset Acquisition

A dedicated acquisition tool (`src/capture.py`) was developed to systematically collect training images. The robot is positioned at a lower scanning height to maximise die resolution (~80--100 px per crop versus ~40 px at the standard height). The tool supports three collection modes:

- **Interactive mode:** each crop is displayed for manual labelling via terminal input.
- **Fixed-label mode:** all crops in a session are assigned the same class label (e.g., `--label J`).
- **Raw mode:** full frames are saved without segmentation for offline processing.

Multiple capture sessions were conducted under varying lighting conditions, die positions, and orientations. The resulting dataset comprises **1,007 images** distributed across six classes:

| Class | 9 | 10 | J | Q | K | A |
|-------|---|----|---|---|---|---|
| **Samples** | 154 | 196 | 152 | 160 | 197 | 148 |

### 3.4 CNN Architecture and Training

#### Network Architecture

The classifier is a three-block convolutional neural network (DiceCNN) designed for lightweight inference:

| Layer | Input Dimensions | Output Dimensions | Parameters |
|-------|-----------------|-------------------|------------|
| Conv2d(3, 32, 3, padding=1) + BatchNorm + ReLU + MaxPool(2) | 3 x 64 x 64 | 32 x 32 x 32 | 960 |
| Conv2d(32, 64, 3, padding=1) + BatchNorm + ReLU + MaxPool(2) | 32 x 32 x 32 | 64 x 16 x 16 | 18,624 |
| Conv2d(64, 128, 3, padding=1) + BatchNorm + ReLU + MaxPool(2) | 64 x 16 x 16 | 128 x 8 x 8 | 74,112 |
| Flatten + Dropout(0.5) + Linear(8192, 256) + ReLU | 8,192 | 256 | 2,097,408 |
| Dropout(0.3) + Linear(256, 6) | 256 | 6 | 1,542 |
| **Total** | | | **~2.2M** |

The input is a 64 x 64 RGB crop normalised to [0, 1]. The output is a six-dimensional logit vector, one per class, with softmax applied at inference time.

#### Training Protocol

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) |
| Loss function | Cross-entropy |
| Epochs | 50 |
| Train/validation split | 80/20 (stratified) |
| Class balancing | Weighted random sampler (inverse class frequency) |
| Data augmentation | Random rotation ($\pm 30\degree$), horizontal flip, colour jitter, affine translation |

The best checkpoint is selected by validation accuracy and exported to ONNX format (opset 11) for deployment with ONNX Runtime. Inference latency is approximately 2--5 ms per crop on a standard laptop CPU.

### 3.5 Poker Hand Evaluation

The evaluator module implements the standard poker dice hand hierarchy:

| Rank | Hand | Description |
|------|------|-------------|
| 1 | Nothing | No matching faces |
| 2 | Pair | Two matching faces |
| 3 | Two Pair | Two distinct pairs |
| 4 | Three of a Kind | Three matching faces |
| 5 | Straight | Five consecutive faces (9-10-J-Q-K or 10-J-Q-K-A) |
| 6 | Full House | Three of a kind + a pair |
| 7 | Four of a Kind (Poker) | Four matching faces |
| 8 | Five of a Kind (Repoker) | All five faces identical |

The evaluation function receives a list of classified face labels, computes frequency counts, and returns the hand name, rank, and a human-readable description (e.g., *"Full House: trio de 9, par de K"*).

![Terminal output with identified hand](assets/media/terminal_identifacion_jugada.png)
*Figure 3. Terminal output showing the hand identification alongside ongoing pick-and-place operations.*

### 3.6 Robotic Pick-and-Place

The physical sorting strategy assigns each distinct face to a dedicated column on the workspace. Dice sharing the same face are stacked vertically within their column. This arrangement provides an immediate visual encoding of the hand structure:

- **Two Pair** (9, 9, J, J, K) produces three columns: two dice in the 9-column, two in the J-column, one in the K-column.
- **Full House** produces two columns (trio and pair).
- **Repoker** produces a single column of five dice.

The pick-and-place routine operates as follows for each die:

1. The gripper is explicitly opened to prevent carry-over from previous operations.
2. The arm returns to the scanning pose and captures a fresh frame.
3. The workspace and dice are re-detected to compensate for positional drift caused by arm movement.
4. The target die is identified by matching face label and proximity to its last known position.
5. The arm approaches, descends, grasps, lifts, translates to the target column, and releases.

Column separation was empirically set to 0.20 normalised units (~3 cm physical space) to prevent gripper collisions with adjacent columns.

![Picking a die](assets/media/2_pickeo.gif)
*Figure 4. The robotic arm executing a pick operation on a single die.*

![Grouping dice by face](assets/media/3_agrupamiento.gif)
*Figure 5. Dice being grouped into columns according to their classified face labels.*

---

## 4. Results

### Classification Performance

The CNN achieves a **validation accuracy of 98.5%** on the held-out set, with only three misclassifications across the entire validation partition. All errors occur between the visually similar classes `9` and `10`, which is consistent with the high degree of visual similarity between these two faces on the physical dice.

![Live classification](assets/media/1_2identificacion.gif)
*Figure 6. Real-time classification on the robot's camera feed. Each die is recognised with high confidence (typically >99%), and the evaluated poker hand is displayed in the overlay.*

### End-to-End Pipeline

The complete system operates reliably in real time, as demonstrated in the following sequence:

1. The robot captures a frame and the vision module detects the workspace and segments all dice.
2. The CNN classifies each crop with high confidence.
3. The evaluator determines the poker hand.
4. Upon issuing the `pick all` command, the robot autonomously sorts all dice into their respective columns.

![Complete pipeline](assets/media/4_final.gif)
*Figure 7. End-to-end execution: workspace detection, dice classification, hand evaluation, and autonomous column-based sorting performed in a single uninterrupted sequence.*

---

## 5. Discussion

The principal insight from this work is the effectiveness of separating the system into **geometric** and **semantic** layers connected through a clean intermediate representation. The normalised workspace coordinates $[0, 1]^2$ serve as this interface: the vision module operates entirely in pixel space, the robot controller operates in physical coordinates, and neither needs knowledge of the other's internal representation.

The CNN component, despite its simplicity (three convolutional blocks, ~2.2M parameters), proves sufficient for this task, which is characterised by a small number of classes with high inter-class visual distinctness. The ONNX export enables deployment without PyTorch as a runtime dependency, reducing the inference overhead to a few milliseconds per crop.

On the robotic manipulation side, the most challenging aspects were not perceptual but physical: ensuring consistent gripper state between operations, calibrating column separation to prevent collisions, and implementing per-pick re-detection to compensate for positional drift. These practical considerations, though straightforward in principle, required iterative empirical tuning.

**Limitations.** The current system assumes a static, flat workspace with controlled lighting. Performance may degrade under significant illumination changes or if the workspace surface is non-planar. The dataset, while sufficient for the controlled laboratory environment, is relatively small by deep learning standards and may not generalise to dice from different manufacturers.

---

## 6. Project Structure

```
computervision-niryo/
├── README.md                       # This document
├── requirements.txt                # Python dependencies
├── assets/
│   ├── dataset/                    # Training images (1,007 samples)
│   │   ├── 9/                      # 154 images
│   │   ├── 10/                     # 196 images
│   │   ├── J/                      # 152 images
│   │   ├── Q/                      # 160 images
│   │   ├── K/                      # 197 images
│   │   └── A/                      # 148 images
│   ├── media/                      # Documentation figures and GIFs
│   └── models/                     # Trained model artefacts
│       ├── best_model.pth          # PyTorch checkpoint
│       ├── dice_classifier.onnx    # ONNX model for inference
│       └── dice_classifier.onnx.data
└── src/
    ├── config.py                   # Centralised configuration parameters
    ├── vision.py                   # Workspace calibration and dice segmentation
    ├── classifier.py               # ONNX-based CNN inference
    ├── evaluator.py                # Poker hand evaluation logic
    ├── robot.py                    # Niryo arm control and coordinate mapping
    ├── poker.py                    # Main application orchestrator
    ├── train.py                    # CNN training and ONNX export pipeline
    ├── capture.py                  # Dataset acquisition tool
    ├── dataset.py                  # Dataset utilities
    ├── main.py                     # Legacy entry point (Lab 1: coloured shapes)
    └── ui.py                       # Legacy UI module (Lab 1)
```

---

## 7. Installation and Usage

### Prerequisites

- Python 3.9+
- Niryo One robotic arm (accessible at the IP configured in `src/config.py`)
- USB or network camera (or the Niryo's built-in camera)

### Setup

```bash
git clone <repository-url>
cd computervision-niryo
pip install -r requirements.txt
```

### Training the CNN (optional)

If retraining the classifier is desired (e.g., with an expanded dataset):

```bash
python src/train.py
```

This will train the DiceCNN for 50 epochs and export the best model to `assets/models/dice_classifier.onnx`.

### Dataset Capture (optional)

To collect additional training images:

```bash
# Interactive labelling mode
python src/capture.py

# Fixed-label mode (all crops labelled as "J")
python src/capture.py --label J

# Low-height capture for higher resolution crops
python src/capture.py --low
```

### Running the System

```bash
python src/poker.py
```

#### Terminal Commands

| Command | Description |
|---------|-------------|
| `help` | Display available commands |
| `scan` | Move arm to scanning pose and refresh detection |
| `home` | Move arm to home position |
| `open` | Open the gripper |
| `select N` | Select die number N |
| `pick` | Pick the selected die and place it in its assigned column |
| `pick all` | Autonomously sort all detected dice |
| `reset` | Clear column assignments for a new round |
| `status` | Display current system state |
| `exit` | Terminate the application |

---

## 8. Conclusions

This work demonstrates the integration of classical computer vision, deep learning, and robotic manipulation into a cohesive system capable of autonomous scene understanding and physical interaction. The key contributions are:

1. **A robust workspace calibration pipeline** using fiducial marker detection and homography estimation that provides a stable normalised coordinate frame for downstream processing.
2. **An edge-based dice segmentation method** that overcomes the challenge of white-on-white detection without relying on colour information.
3. **A lightweight CNN classifier** achieving 98.5% accuracy on six poker dice face classes, deployed efficiently via ONNX Runtime.
4. **A column-based sorting strategy** with per-pick re-detection that provides physical visualisation of the evaluated poker hand while compensating for positional drift.

The design principle of maintaining a **clean intermediate representation** (the normalised workspace) proved essential for managing system complexity, enabling each module to be developed, tested, and debugged independently before integration.

---

## 9. References

- Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*.
- Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.
- ONNX Runtime. https://onnxruntime.ai/
- Niryo. *PyNiryo Documentation*. https://docs.niryo.com/dev/pyniryo/index.html
- Canny, J. (1986). A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679--698.
- Duda, R. O., & Hart, P. E. (1972). Use of the Hough Transformation to Detect Lines and Curves in Pictures. *Communications of the ACM*, 15(1), 11--15.
