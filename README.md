# Stereo Vision – Approximate Depth Estimation

## Project Overview
This project demonstrates **approximate sparse depth estimation** using a pair of stereo images (`Left.jpg` and `Right.jpg`) captured with a smartphone camera. Two images of a scene were taken from horizontally separated positions with a baseline distance of **20.0 cm**, targeting an object placed at a reference distance of **100.0 cm**.

Instead of computing a dense disparity map across the entire image, this project uses an interactive, feature-based method (SIFT + RANSAC) to measure the horizontal shift (**disparity**) of a selected target object and calculate its approximate distance.

> **Note on Accuracy**: The calculated depth values are approximate. The camera was not formally calibrated (no checkerboard intrinsic matrix), and the photos were taken handheld without formal stereo rectification.

---

## Objectives
- Capture stereo image pairs of a target object with a known baseline ($B = 20.0\text{ cm}$).
- Interactively select a Region of Interest (ROI) to isolate the target object.
- Detect distinctive keypoints and find feature correspondences using SIFT and Lowe's ratio test.
- Filter false correspondences using RANSAC and remove vertical alignment errors with a common vertical-offset threshold.
- Estimate the object's approximate depth using the stereo depth formula $Z = (f \times B) / d$ and compare it with the measured reference distance ($100.0\text{ cm}$).

---

## Tools Used
- **Python**: Core programming language.
- **OpenCV (`cv2`)**: SIFT feature detection, descriptor matching (BFMatcher), RANSAC Fundamental Matrix estimation, and interactive ROI selection (`cv2.selectROI`).
- **NumPy**: Vectorized coordinate calculations, masking, and median disparity estimation.
- **Matplotlib**: Plotting disparity distribution histograms.

---

## How It Works

### 1. Stereo Vision Principle
When an object is photographed from two horizontally separated viewpoints, it appears at slightly different horizontal positions in the two images. This difference in horizontal coordinates is called **disparity** ($d$):
- **Closer objects** shift more between views $\rightarrow$ **higher disparity**.
- **Farther objects** shift less between views $\rightarrow$ **lower disparity**.

Using simple pinhole stereo geometry, depth ($Z$) is calculated using the formula:

$$Z = \frac{f \times B}{d}$$

Where:
- **$Z$**: Estimated distance to the object in centimeters.
- **$f$**: Approximate focal length in pixels ($2699.32\text{ px}$, estimated from the phone camera's 26 mm 35mm-equivalent focal length and image diagonal).
- **$B$**: Stereo baseline ($20.0\text{ cm}$, the distance between camera positions).
- **$d$**: Median horizontal disparity in pixels.

### 2. Step-by-Step Pipeline
1. **Load & Scale**: Both images ($4096 \times 1844$) are loaded and resized by a scale factor of $0.5$ ($2048 \times 922$) for faster processing.
2. **Interactive ROI Selection**: OpenCV windows open so the user can draw a bounding box around the target in both the left and right images.
3. **SIFT Feature Matching**: SIFT extracts keypoints within the selected region. Matches are found with a Brute-Force Matcher and filtered using Lowe's ratio test (threshold $0.75$).
4. **RANSAC Geometric Filtering**: Outlier matches are removed by computing the Fundamental Matrix with RANSAC ($1.5\text{ px}$ threshold).
5. **Auto Order Correction**: If the median horizontal shift is negative, the script automatically swaps the left and right frames to ensure positive disparity ($d > 0$).
6. **Heuristic Vertical Filtering**: Handheld photos have minor vertical shifts. The script estimates the common vertical offset using $\text{median}(\Delta y)$ and retains matches where $|\Delta y - y_{\text{offset}}| < 8\text{ pixels}$.
7. **Depth Estimation**: Disparity values are scaled back to original resolution, and the median disparity is used in $Z = (f \times B) / d$.
8. **Save Results**: Outputs are saved to disk (`feature_matches.jpg`, `disparity_graph.png`, and `experiment_summary.txt`).

---

## Experiments and Results

The pipeline was evaluated across three different ROI selections using the same baseline ($B = 20.0\text{ cm}$) and reference distance ($Z_{\text{ref}} = 100.0\text{ cm}$):
- **Experiment 1: Circular fan head** (isolates the circular head of the fan)
- **Experiment 2: Complete fan** (includes circular head, buttons, stand, and base)
- **Experiment 3: Complete image with background** (entire unconstrained frame)

### Results Table

| Experiment | Target Region | RANSAC Inliers | Valid Matches | Median Disparity (px) | Residual Vertical Error (px) | Estimated Depth (cm) | Reference Distance (cm) | Absolute Error (cm) | Percentage Error (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Experiment 1** | Circular fan head | 68 | 65 | 566.22 | 2.96 | 95.35 | 100.00 | 4.65 | 4.65% |
| **Experiment 2** | Complete fan | 81 | 76 | 560.99 | 5.04 | 96.23 | 100.00 | 3.77 | 3.77% |
| **Experiment 3** | Complete image with background | 87 | 79 | 556.43 | 6.98 | 97.02 | 100.00 | 2.98 | 2.98% |

### Discussion of Results:
- **Experiment 1 (Circular Fan Head)**: Strictly focuses on the fan head. It gives the lowest residual vertical error ($2.96\text{ px}$), showing consistent matches on the primary target.
- **Experiment 2 (Complete Fan)**: Includes features across the stand and base at varying depth levels, resulting in a median disparity of $560.99\text{ px}$ and depth of $96.23\text{ cm}$.
- **Experiment 3 (Complete Image)**: Shows the lowest percentage error ($2.98\%$), but this **does not mean it is a better fan depth measurement**. Matches are scattered across the walls and floor, and background features pull the overall disparity distribution closer to $100\text{ cm}$ by coincidence. Experiment 1 is the most accurate representation of the fan head itself.

---

## Result Images

### Experiment 1: Circular Fan Head (`results(fan)/`)
| Feature Matches | Disparity Distribution |
| :---: | :---: |
| ![Experiment 1 Matches](results(fan)/feature_matches.jpg) | ![Experiment 1 Disparity](results(fan)/disparity_graph.png) |

### Experiment 2: Complete Fan (`results(complete_fan)/`)
| Feature Matches | Disparity Distribution |
| :---: | :---: |
| ![Experiment 2 Matches](results(complete_fan)/feature_matches.jpg) | ![Experiment 2 Disparity](results(complete_fan)/disparity_graph.png) |

### Experiment 3: Complete Image with Background (`results(complete_image)/`)
| Feature Matches | Disparity Distribution |
| :---: | :---: |
| ![Experiment 3 Matches](results(complete_image)/feature_matches.jpg) | ![Experiment 3 Disparity](results(complete_image)/disparity_graph.png) |

---

## How to Run

### 1. Requirements
Install the required Python packages:
```bash
pip install opencv-python numpy matplotlib
```

### 2. Execution
Run the stereo vision script from the project folder:
```bash
python stereo_vision.py
```

### 3. Interactive Steps
1. An OpenCV window titled **"Select Fan - Left Image"** will appear. Click and drag a box around the object you want to measure, then press **`ENTER`** or **`SPACE`**.
2. A second window titled **"Select Fan - Right Image"** will appear. Select the same region in the right image, then press **`ENTER`** or **`SPACE`**.
3. The script will print the disparity and depth results in the terminal and write three output files into the `results/` folder:
   - `feature_matches.jpg`: Visualization of matched keypoints.
   - `disparity_graph.png`: Disparity distribution histogram.
   - `experiment_summary.txt`: Text summary of all parameters and results.

> **Note**: The script saves to `results/` by default. To preserve multiple experiment runs, rename or copy the `results/` folder before running another trial.

---

## Limitations
- **Approximate Focal Length**: Focal length ($f \approx 2699.32\text{ px}$) is derived from the phone's 35mm-equivalent sensor diagonal, not from formal checkerboard camera calibration.
- **No True Stereo Rectification**: Images were captured handheld without rigid mounting. A heuristic vertical filter is used, but epipolar lines were not mathematically rectified to horizontal scanlines.
- **Sparse Depth Only**: The script computes a single median depth for the selected region using matched keypoints; it does not generate a full 3D point cloud or dense depth map.
- **Interactive Manual Input**: The quality of feature matches depends on the user correctly selecting matching bounding boxes in both images.
