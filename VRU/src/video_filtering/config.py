import os

# --- Path configuration ---
# Project root directory
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = "/home/24068286g/UString"

# Input data paths
CCD_ROOT_PATH = os.path.join(ROOT_DIR, 'data', 'crash')
ANNOTATION_FILE = os.path.join(CCD_ROOT_PATH, 'videos', 'Crash-1500.txt')
NPZ_DIR = os.path.join(CCD_ROOT_PATH, 'yolo_features', 'positive')

# Output data path
OUTPUT_DIR = os.path.join(ROOT_DIR, 'VRU', 'output2')

# --- Output files for the three filtering strategies ---
# Strategy 1: filter using Scene Complexity only
FILTERED_COMPLEXITY_ONLY = os.path.join(OUTPUT_DIR, 'filtered_complexity_only.json')

# Strategy 2: filter using Dynamic Change only
FILTERED_DYNAMIC_ONLY = os.path.join(OUTPUT_DIR, 'filtered_dynamic_only.json')

# Strategy 3: joint filtering on both dimensions (recommended)
FILTERED_COMBINED = os.path.join(OUTPUT_DIR, 'filtered_combined.json')

# Consolidated report output
REPORT_OUTPUT = os.path.join(OUTPUT_DIR, 'filtering_report.txt')

# Comparative analysis output (new)
COMPARISON_REPORT = os.path.join(OUTPUT_DIR, 'strategy_comparison.json')


# --- Filtering criteria configuration ---
"""
Metric selection rationale:

This study uses two independent video quality metrics. Their selection is based on the following theoretical and empirical evidence:

1. Scene Complexity
   - Theoretical basis: traffic accidents usually involve multiple participants and complex road environments
   - Computation: maximum object count detected within the time window
   - Empirical support: complex scenes are more likely to contain rich interaction information and key events
   - Statistical profile: right-skewed, non-normal distribution (Mean=4.58, Std=2.80, Skewness=0.707)

2. Dynamic Change
   - Theoretical basis: accidents involve abrupt motion-state changes
   - Computation: maximum inter-frame distance between feature vectors (global P95 normalization)
   - Empirical support: videos with stronger dynamic changes are more likely to contain critical moments
   - Note: two-stage global normalization avoids the loss of separability caused by local normalization


Why choose these two metrics instead of others?

Excluded candidate metrics:
- Speed / acceleration: requires multi-frame tracking and calibration, which is expensive and sensitive to occlusion
- Collision angle: requires accurate 3D pose estimation and can be error-prone
- Weather / lighting: weakly related to the accident content and mainly affects detection quality
- Video duration: not directly related to content quality and varies widely across datasets
- Resolution / frame rate: technical rather than content-based, and is already reflected indirectly in detection confidence

Advantages of these two metrics:
1. They depend only on object detection results and require no extra annotation or tracking
2. They are computationally efficient and scale to large datasets
3. Their physical meaning is clear and easy to explain and validate
4. They are complementary in space and time, providing a more complete assessment of video quality
5. They are objective and do not depend on subjective judgment
"""

# Object category IDs (based on COCO standard - VRU-specific labels deprecated)

# Time window: number of frames to include before and after the accident frame
# 30 frames is roughly 1 second at 30 fps, which is enough to capture the key motion before and after the accident
TIME_WINDOW_FRAMES = 30

# --- Threshold configuration ---
# Confidence threshold: only detections above this confidence are considered
CONFIDENCE_THRESHOLD = 0.5

# Strategy 1: Dynamic Change threshold (based on inter-frame distance of average features)
# Selection basis: marginal gain analysis
# Note: Dynamic Change is computed from the average feature vector of all high-confidence detections
DYNAMIC_CHANGE_THRESHOLD = 0.6

# Strategy 2: Scene Complexity threshold (single-frame peak object count)
# Selection basis: marginal gain analysis
# Note: Scene Complexity = max number of high-confidence detections in any frame within the window
COMPLEXITY_THRESHOLD = 6

# --- Output configuration ---
# Output files for the three strategies are defined above