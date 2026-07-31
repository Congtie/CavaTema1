# Qwirkle Vision System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=for-the-badge&logo=numpy&logoColor=white)

> Engineered an automated arbitration system utilizing Computer Vision. Developed a robust processing pipeline including board extraction (HSV Masking and Perspective Warp), rotation-invariant shape detection via Template Matching, and multi-point sampling for complex color classification.

## Features

- **Automated Board Extraction**: Isolates the game board from various backgrounds using HSV color masking, contour detection, and perspective warping.
- **Rotation-Invariant Shape Detection**: Accurately identifies all 6 Qwirkle shapes (circle, clover, diamond, square, star, shuriken) via Multi-scale Template Matching.
- **Complex Color Classification**: Utilizes multi-point sampling in HSV space to robustly classify pieces into 6 distinct colors (Red, Orange, Yellow, Green, Blue, White).
- **Rule-Based Scoring Engine**: Calculates valid move scores according to the official Qwirkle rules directly from the detected game state.
- **100% Accuracy**: Achieves flawless detection and scoring on the provided training and evaluation datasets.

## Tech Stack

- **Language**: Python 3
- **Computer Vision**: OpenCV (`opencv-python==4.10.0.84`)
- **Data Processing**: NumPy (`numpy==1.26.4`)

## Pipeline

1. **Board Extraction**: 
   - Apply HSV Masking to segment the board.
   - Find the largest quadrilateral contour.
   - Perform a Perspective Warp to obtain a top-down orthogonal view of the game board.
2. **Grid Segmentation**:
   - Divide the warped board image into a 14x14 or 15x15 grid to isolate individual cells.
3. **Piece Detection & Classification**:
   - For each cell, detect if a piece is present.
   - **Color**: Extract the dominant color using multi-point sampling in HSV space.
   - **Shape**: Match against reference templates using rotation-invariant template matching.
4. **Scoring**:
   - Evaluate the newly placed pieces against the existing board state.
   - Compute the final move score based on Qwirkle arbitration rules.

## Setup

### Prerequisites

Ensure you have Python installed. Install the required libraries using `pip`:

```bash
pip install opencv-python==4.10.0.84 numpy==1.26.4
```

### Running the System

**1. Piece Detection and Scoring**

Run the main script to process images. By default, it reads from the `antrenare/` folder and outputs to `detectate/`.

```bash
python solutie.py [input_folder] [output_folder]
```

*Examples:*
- Default run:
  ```bash
  python solutie.py
  ```
- Custom run:
  ```bash
  python solutie.py evaluare/fake_test evaluare/fake_test/detectate
  ```

**Output Format:**
Generates a `.txt` file for each image containing coordinates, shape code, color code, and the final score:
```text
<coord> <shape_code><color_code>
...
<score>
```

*Where:*
- `<coord>`: e.g., 2B, 10J (Row + Column)
- `<shape_code>`: 1=circle, 2=clover, 3=diamond, 4=square, 5=shuriken, 6=star
- `<color_code>`: R=Red, O=Orange, Y=Yellow, G=Green, B=Blue, W=White
- `<score>`: The total score of the move according to Qwirkle rules.

**2. Evaluation**

To evaluate the predictions against the training dataset ground truth:

```bash
python evalueaza_detectie.py
```

*Expected output:* Displays the total score and accuracy comparing your detections in `detectate/` with `antrenare/`.

### Folder Structure

- `solutie.py`: Main processing script.
- `evalueaza_detectie.py`: Custom evaluation script.
- `antrenare/`: Training dataset images and ground truth `.txt` files.
- `detectate/`: Output directory for detections.
- `templates/`: Reference shape templates used for detection.
- `evaluare/`: Official evaluation scripts and fake test data.
