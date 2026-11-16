
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================================================
# 1. EXPERIMENT PARAMETERS
# ==================================================

LEFT_IMAGE = "Left.jpg"
RIGHT_IMAGE = "Right.jpg"

BASELINE = 20.0
REFERENCE_DISTANCE = 100.0
FOCAL_LENGTH_EQ = 26.0

SCALE = 0.5
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# 2. LOAD IMAGES
# ==================================================

left = cv2.imread(LEFT_IMAGE)
right = cv2.imread(RIGHT_IMAGE)

if left is None or right is None:
    raise FileNotFoundError("Check Left.jpg and Right.jpg.")

if left.shape != right.shape:
    raise ValueError("Image dimensions must be equal.")

height, width = left.shape[:2]

focal_length = (
    FOCAL_LENGTH_EQ
    * np.sqrt(width**2 + height**2)
    / np.sqrt(36**2 + 24**2)
)

left = cv2.resize(left, None, fx=SCALE, fy=SCALE)
right = cv2.resize(right, None, fx=SCALE, fy=SCALE)

gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

print("\n========== STEREO VISION ==========")
print("Image resolution:", width, "x", height)
print("Approximate focal length:", round(focal_length, 2), "pixels")


# ==================================================
# 3. SELECT FAN REGION (ROI)
# ==================================================

print("\nSelect ONLY the circular fan head in each image.")
print("Exclude the wall, floor and surrounding background.")
print("Press ENTER after selecting each region.")

roi_left = cv2.selectROI(
    "Select Fan - Left Image",
    left,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

roi_right = cv2.selectROI(
    "Select Fan - Right Image",
    right,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()


def create_mask(image, roi):

    x, y, w, h = map(int, roi)

    if w == 0 or h == 0:
        raise ValueError("You must select a fan region in both images.")

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    mask[y:y+h, x:x+w] = 255

    return mask


mask_left = create_mask(left, roi_left)
mask_right = create_mask(right, roi_right)


# ==================================================
# 4. SIFT FEATURE DETECTION AND MATCHING
# ==================================================

def match_features(img1, img2, mask1, mask2):

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=3000)

    kp1, des1 = sift.detectAndCompute(gray1, mask1)
    kp2, des2 = sift.detectAndCompute(gray2, mask2)

    if des1 is None or des2 is None:
        raise RuntimeError("Not enough fan features detected.")

    print("\nLeft fan keypoints:", len(kp1))
    print("Right fan keypoints:", len(kp2))

    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good.append(m)

    print("Good matches:", len(good))

    if len(good) < 8:
        raise RuntimeError("Insufficient matching fan features.")

    p1 = np.float32([
        kp1[m.queryIdx].pt for m in good
    ])

    p2 = np.float32([
        kp2[m.trainIdx].pt for m in good
    ])

    F, ransac_mask = cv2.findFundamentalMat(
        p1,
        p2,
        cv2.FM_RANSAC,
        1.5,
        0.99
    )

    if F is None or ransac_mask is None:
        raise RuntimeError("RANSAC calculation failed.")

    inlier_indices = np.flatnonzero(
        ransac_mask.ravel().astype(bool)
    )

    p1 = p1[inlier_indices]
    p2 = p2[inlier_indices]

    inlier_matches = [
        good[i] for i in inlier_indices
    ]

    if len(p1) < 8:
        raise RuntimeError("Not enough RANSAC inliers.")

    dx = p1[:, 0] - p2[:, 0]
    dy = p1[:, 1] - p2[:, 1]

    return kp1, kp2, inlier_matches, p1, p2, dx, dy


# ==================================================
# 5. AUTOMATIC IMAGE ORDER CORRECTION
# ==================================================

result = match_features(
    left, right,
    mask_left, mask_right
)

kp1, kp2, matches, p1, p2, dx, dy = result

if np.median(dx) < 0:

    print("\nReversing image order for positive disparity.")

    left, right = right, left
    gray_left, gray_right = gray_right, gray_left
    mask_left, mask_right = mask_right, mask_left

    result = match_features(
        left, right,
        mask_left, mask_right
    )

    kp1, kp2, matches, p1, p2, dx, dy = result

print("\nRANSAC inliers:", len(matches))


# ==================================================
# 6. FILTER MATCHES AND CALCULATE DISPARITY
# ==================================================

# Correct for a common vertical offset between images
vertical_offset = np.median(dy)

valid = (
    (dx > 0) &
    (np.abs(dy - vertical_offset) < 8)
)

disparities = dx[valid] / SCALE

selected_matches = [
    m for m, keep in zip(matches, valid) if keep
]

if len(disparities) == 0:
    raise RuntimeError("No valid fan disparities found.")

median_disparity = np.median(disparities)

median_vertical = np.median(
    np.abs(dy[valid] - vertical_offset)
) / SCALE

print("\n========== DISPARITY RESULTS ==========")
print("Valid fan matches:", len(disparities))
print("Median horizontal disparity:", round(median_disparity, 2), "pixels")
print("Common vertical offset:", round(vertical_offset / SCALE, 2), "pixels")
print("Residual vertical error:", round(median_vertical, 2), "pixels")


# ==================================================
# 7. FEATURE MATCHING VISUALIZATION
# ==================================================

match_image = cv2.drawMatches(
    left,
    kp1,
    right,
    kp2,
    selected_matches[:80],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    f"{OUTPUT_DIR}/feature_matches.jpg",
    match_image
)


# ==================================================
# 8. SPARSE DEPTH ESTIMATION
# ==================================================

depth = (
    focal_length * BASELINE
    / median_disparity
)

error = abs(depth - REFERENCE_DISTANCE)

percentage_error = (
    error / REFERENCE_DISTANCE
) * 100

print("\n========== DEPTH ESTIMATION ==========")
print("Baseline:", BASELINE, "cm")
print("Reference distance:", REFERENCE_DISTANCE, "cm")
print("Estimated depth:", round(depth, 2), "cm")
print("Absolute error:", round(error, 2), "cm")
print("Percentage error:", round(percentage_error, 2), "%")


# ==================================================
# 9. DISPARITY GRAPH
# ==================================================

plt.figure(figsize=(9, 5))

plt.hist(
    disparities,
    bins=15,
    edgecolor="black"
)

plt.axvline(
    median_disparity,
    linestyle="--",
    linewidth=2,
    label=f"Median = {median_disparity:.2f} px"
)

plt.xlabel("Horizontal Disparity (pixels)")
plt.ylabel("Number of Fan Matches")
plt.title("Fan - Disparity Distribution")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/disparity_graph.png",
    dpi=150
)

plt.close()


# ==================================================
# 10. SAVE EXPERIMENT SUMMARY
# ==================================================

summary = f"""
STEREO VISION - LAB EVALUATION

Camera: iQOO Z10R 5G
Resolution: {width} x {height}

EXPERIMENT PARAMETERS
---------------------
Baseline: {BASELINE} cm
Reference Distance: {REFERENCE_DISTANCE} cm
Equivalent Focal Length: {FOCAL_LENGTH_EQ} mm
Approximate Focal Length: {focal_length:.2f} pixels

FEATURE MATCHING
----------------
RANSAC Inliers: {len(matches)}
Valid Fan Matches: {len(disparities)}

DISPARITY
---------
Median Horizontal Disparity: {median_disparity:.2f} pixels
Median Residual Vertical Error: {median_vertical:.2f} pixels

DEPTH ESTIMATION
----------------
Estimated Depth: {depth:.2f} cm
Reference Distance: {REFERENCE_DISTANCE:.2f} cm
Absolute Error: {error:.2f} cm
Percentage Error: {percentage_error:.2f} %

METHOD
------
SIFT Feature Detection
Brute Force Matching
RANSAC Outlier Removal
Fan ROI Filtering
Sparse Disparity Calculation
Depth Estimation: Z = fB/d

NOTE
----
Approximate sparse depth estimation.
Camera calibration has not been performed.
"""

with open(
    f"{OUTPUT_DIR}/experiment_summary.txt",
    "w"
) as file:
    file.write(summary)


# ==================================================
# 11. FINAL OUTPUT
# ==================================================

print("\n====================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("====================================")
print("Results saved in:", OUTPUT_DIR)
print("1. feature_matches.jpg")
print("2. disparity_graph.png")
print("3. experiment_summary.txt")
