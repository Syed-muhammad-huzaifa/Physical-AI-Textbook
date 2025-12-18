---
title: "Visual SLAM and Localization for Humanoid Robotics"
sidebar_position: 3
description: "Comprehensive guide to visual SLAM and localization techniques for humanoid robots, including monocular, stereo, and RGB-D approaches for navigation and mapping"
tags: [vslam, localization, humanoid, robotics, computer-vision, mapping, navigation]
---

# Visual SLAM and Localization for Humanoid Robotics

Visual Simultaneous Localization and Mapping (VSLAM) is a critical capability for humanoid robots that need to navigate and operate in unknown environments. This chapter explores the principles, techniques, and practical implementations of VSLAM systems specifically designed for humanoid robotics applications.

## Learning Objectives

By the end of this chapter, you will be able to:

1. Understand the fundamental principles of Visual SLAM and its applications in humanoid robotics
2. Implement monocular, stereo, and RGB-D SLAM algorithms for humanoid navigation
3. Design robust localization systems that handle humanoid-specific challenges
4. Evaluate and optimize VSLAM performance for real-time humanoid applications
5. Integrate VSLAM systems with humanoid control and planning frameworks
6. Address challenges in dynamic environments with moving objects and changing lighting

## Introduction

Visual SLAM enables humanoid robots to simultaneously build a map of their environment while tracking their position within it using visual sensors. For humanoid robots, VSLAM is particularly important because:

- **3D Environment Understanding**: Humanoid robots need to navigate in 3D space with complex obstacles
- **Human-like Perception**: Vision is the primary sensory modality for humans, making it natural for humanoid robots
- **Cognitive Mapping**: Visual maps align with human spatial reasoning and navigation strategies
- **Social Navigation**: Visual SLAM helps humanoid robots navigate safely around humans
- **Manipulation Planning**: Accurate visual maps are essential for planning interaction with objects

Traditional SLAM approaches using laser range finders work well but are limited in their ability to understand the semantic content of environments. Visual SLAM, on the other hand, can capture rich visual information that enables more sophisticated understanding of the environment.

## Prerequisites

Before diving into VSLAM for humanoid robots, ensure you have:

- Strong understanding of computer vision fundamentals (feature detection, matching, camera models)
- Experience with ROS2 and robot navigation systems
- Knowledge of 3D geometry, rotations, and transformations
- Familiarity with optimization techniques and filtering methods (EKF, particle filters)
- Basic understanding of graph optimization and bundle adjustment
- Python and C++ programming experience for SLAM implementations

## Theory and Concepts

### Visual SLAM Fundamentals

Visual SLAM combines computer vision and SLAM algorithms to solve the dual problem of localization and mapping. The core components include:

**Front-end**: Visual odometry and feature tracking
**Back-end**: Optimization and map refinement
**Loop closure**: Detecting revisited locations to correct drift
**Mapping**: Creating and maintaining the environment representation

The visual SLAM pipeline typically follows this sequence:
1. Feature extraction and matching
2. Motion estimation
3. Bundle adjustment
4. Map management
5. Loop closure detection

```mermaid
graph TD
    A[Visual Input] --> B[Feature Detection]
    B --> C[Feature Matching]
    C --> D[Pose Estimation]
    D --> E[Bundle Adjustment]
    E --> F[Map Building]
    F --> G{Loop Closure?}
    G -->|Yes| H[Drift Correction]
    G -->|No| I[Map Update]
    H --> J[Optimized Map]
    I --> J
    J --> K[Local Path Planning]
    K --> L[Humanoid Navigation]

    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style E fill:#e8f5e8
    style H fill:#fff3e0
    style L fill:#e0f2f1
</graph>

### VSLAM Approaches for Humanoid Robots

#### Monocular SLAM
Monocular SLAM uses a single camera and is particularly challenging because depth cannot be directly measured. Key characteristics:

- **Scale Ambiguity**: Only scale changes can be recovered, not absolute scale
- **Initialization**: Requires motion to estimate initial depth
- **Drift**: Tends to accumulate errors over time
- **Advantages**: Lightweight sensors, low computational requirements
- **Disadvantages**: Scale uncertainty, potential for failure in textureless areas

For humanoid robots, monocular SLAM can be enhanced with prior knowledge about human environments (manhattan worlds, planar surfaces).

#### Stereo SLAM
Stereo SLAM uses stereo camera pairs to provide real-time depth estimates:

- **Metric Scale**: Provides true metric scale recovery
- **Real-time Performance**: Depth available for each frame
- **Dense Maps**: Can generate dense 3D point clouds
- **Disadvantages**: Higher computational cost, limited range accuracy

Stereo SLAM is well-suited for humanoid robots that need accurate metric information for navigation and manipulation.

#### RGB-D SLAM
RGB-D SLAM uses RGB-D sensors that provide color and depth information:

- **Dense Information**: Rich visual and depth data
- **Robust Tracking**: Depth provides reliable correspondence
- **Real-time Dense Mapping**: Creates detailed 3D maps
- **Disadvantages**: Limited range, sensitivity to lighting conditions

RGB-D SLAM is particularly effective for indoor humanoid applications where depth sensors operate well.

### Humanoid-Specific Challenges

Humanoid robots face unique challenges in VSLAM:

**Bipedal Motion**: Humanoid walking creates complex, non-smooth motion patterns that differ from wheeled robots
**Height Variations**: Standing, sitting, or crouching changes the viewpoint dramatically
**Body Occlusions**: Robot's own body may occlude the camera view
**Social Navigation**: Need to consider humans as dynamic obstacles
**Multi-floor Navigation**: Stairs and elevators require 3D mapping capabilities

### Feature-Based vs. Direct Methods

**Feature-based Methods**: Extract and track distinctive features (SIFT, ORB, etc.)
- Advantages: Robust to lighting changes, well-understood
- Disadvantages: May fail in textureless environments

**Direct Methods**: Use all pixel intensities for tracking
- Advantages: Works in low-texture environments
- Disadvantages: Sensitive to lighting changes, computationally expensive

For humanoid robots, feature-based methods are often preferred due to their robustness, but hybrid approaches can provide the best performance.

### Loop Closure Detection

Loop closure is crucial for correcting drift in VSLAM systems:

- **Appearance-based**: Compare current view with stored keyframes
- **Place recognition**: Use global scene descriptors
- **Geometric verification**: Confirm matches geometrically
- **Optimization**: Update the map graph when loops are detected

For humanoid robots, loop closure can be enhanced with semantic information about human environments (rooms, corridors, landmarks).

## Practical Implementation

### 1. Visual SLAM System for Humanoid Robots

Let's implement a visual SLAM system suitable for humanoid robots:

```python
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import threading
from queue import Queue
import time

@dataclass
class Pose:
    """Represents a 6DOF pose with position and orientation."""
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [qx, qy, qz, qw] quaternion

class Keyframe:
    """Represents a keyframe in the SLAM system."""
    def __init__(self,
                 image: np.ndarray,
                 pose: Pose,
                 features: np.ndarray,
                 descriptors: np.ndarray,
                 timestamp: float):
        self.image = image
        self.pose = pose
        self.features = features  # [N, 2] feature coordinates
        self.descriptors = descriptors  # [N, 128] for ORB
        self.timestamp = timestamp
        self.id = int(timestamp * 1000)  # Simple ID based on timestamp

class MapPoint:
    """Represents a 3D point in the map."""
    def __init__(self,
                 position: np.ndarray,
                 descriptor: np.ndarray,
                 observations: List[Tuple[int, int]] = None):
        self.position = position  # [x, y, z]
        self.descriptor = descriptor  # Descriptor for matching
        self.observations = observations or []  # [(keyframe_id, feature_idx)]
        self.id = id(self)  # Unique identifier

class VisualSLAM:
    """Visual SLAM system optimized for humanoid robots."""

    def __init__(self,
                 camera_matrix: np.ndarray,
                 distortion_coeffs: np.ndarray,
                 feature_detector_type: str = "ORB"):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.feature_detector_type = feature_detector_type

        # Initialize feature detector
        if feature_detector_type == "ORB":
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unsupported feature detector: {feature_detector_type}")

        # SLAM state
        self.keyframes: List[Keyframe] = []
        self.map_points: List[MapPoint] = []
        self.current_pose = Pose(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        )

        # Tracking state
        self.is_initialized = False
        self.prev_image = None
        self.prev_features = None
        self.prev_descriptors = None

        # Loop closure detection
        self.loop_detector = LoopClosureDetector()

        # For threading
        self.slam_lock = threading.Lock()

    def process_frame(self, image: np.ndarray) -> Pose:
        """Process a new image frame and return estimated pose."""
        with self.slam_lock:
            # Extract features
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            features, descriptors = self.extract_features(gray)

            if len(features) < 50:  # Not enough features
                return self.current_pose

            if not self.is_initialized:
                # Initialize with first frame
                self.initialize_map(image, gray, features, descriptors)
                return self.current_pose
            else:
                # Track motion and update map
                success, pose_change = self.track_frame(
                    gray, features, descriptors
                )

                if success:
                    # Update current pose
                    self.current_pose = self.compose_poses(
                        self.current_pose, pose_change
                    )

                    # Add keyframe if significant motion
                    if self.should_add_keyframe(pose_change):
                        self.add_keyframe(image, gray, features, descriptors)

                        # Optimize map if needed
                        self.optimize_map()

                        # Check for loop closures
                        self.detect_loop_closure()

                # Store for next frame
                self.prev_image = gray
                self.prev_features = features
                self.prev_descriptors = descriptors

                return self.current_pose

    def extract_features(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and descriptors from an image."""
        keypoints = self.detector.detect(gray_image, None)
        if keypoints:
            keypoints, descriptors = self.detector.compute(gray_image, keypoints)
            if descriptors is not None:
                features = np.array([kp.pt for kp in keypoints])
                return features, descriptors
        return np.array([]), np.array([])

    def initialize_map(self, image: np.ndarray, gray: np.ndarray,
                      features: np.ndarray, descriptors: np.ndarray):
        """Initialize the SLAM system with the first keyframe."""
        initial_pose = Pose(
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        keyframe = Keyframe(image, initial_pose, features, descriptors, time.time())
        self.keyframes.append(keyframe)

        self.is_initialized = True
        self.prev_image = gray
        self.prev_features = features
        self.prev_descriptors = descriptors

    def track_frame(self, gray: np.ndarray, features: np.ndarray,
                   descriptors: np.ndarray) -> Tuple[bool, Pose]:
        """Track motion between current and previous frame."""
        if self.prev_descriptors is None or len(self.prev_descriptors) == 0:
            return False, self.current_pose

        # Match features between current and previous frames
        matches = self.matcher.knnMatch(
            self.prev_descriptors, descriptors, k=2
        )

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 20:  # Not enough good matches
            return False, self.current_pose

        # Extract matched points
        prev_pts = np.float32([
            self.prev_features[m.queryIdx] for m in good_matches
        ]).reshape(-1, 1, 2)
        curr_pts = np.float32([
            features[m.trainIdx] for m in good_matches
        ]).reshape(-1, 1, 2)

        # Estimate motion using Essential matrix
        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts, self.camera_matrix,
            method=cv2.RANSAC, threshold=1.0
        )

        if E is None or E.shape[0] != 3:
            return False, self.current_pose

        # Extract rotation and translation
        _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, self.camera_matrix)

        # Convert to pose
        rotation_quat = self.rotation_matrix_to_quaternion(R)
        translation = t.flatten()

        # Create pose change
        pose_change = Pose(position=translation, rotation=rotation_quat)

        return True, pose_change

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to quaternion."""
        # Using the standard algorithm to convert rotation matrix to quaternion
        trace = np.trace(R)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return np.array([qx, qy, qz, qw])

    def compose_poses(self, pose1: Pose, pose2: Pose) -> Pose:
        """Compose two poses: result = pose1 * pose2."""
        # Compose positions
        # R1 * t2 + t1
        composed_pos = (
            self.quaternion_rotate_vector(pose1.rotation, pose2.position) +
            pose1.position
        )

        # Compose rotations (Hamilton product)
        composed_rot = self.quaternion_multiply(pose1.rotation, pose2.rotation)

        return Pose(position=composed_pos, rotation=composed_rot)

    def quaternion_rotate_vector(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q."""
        # Convert vector to quaternion
        v_quat = np.array([v[0], v[1], v[2], 0.0])

        # Conjugate of q
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])

        # Rotate: q * v * q_conj (Hamilton product)
        temp = self.quaternion_multiply(q, v_quat)
        result = self.quaternion_multiply(temp, q_conj)

        return result[:3]

    def quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([x, y, z, w])

    def should_add_keyframe(self, pose_change: Pose) -> bool:
        """Determine if a new keyframe should be added."""
        # Check translation and rotation thresholds
        trans_magnitude = np.linalg.norm(pose_change.position)
        rotation_magnitude = 2 * np.arccos(abs(pose_change.rotation[3]))  # Angle from quaternion

        # Humanoid-specific thresholds (can be tuned)
        trans_threshold = 0.5  # meters
        rot_threshold = 0.3    # radians (~17 degrees)

        return (trans_magnitude > trans_threshold or
                rotation_magnitude > rot_threshold)

    def add_keyframe(self, image: np.ndarray, gray: np.ndarray,
                    features: np.ndarray, descriptors: np.ndarray):
        """Add a new keyframe to the map."""
        keyframe = Keyframe(
            image=image,
            pose=self.current_pose,
            features=features,
            descriptors=descriptors,
            timestamp=time.time()
        )

        self.keyframes.append(keyframe)

    def optimize_map(self):
        """Optimize the map using bundle adjustment."""
        # Simplified optimization - in practice, this would use
        # proper graph optimization libraries like g2o or Ceres
        if len(self.keyframes) < 2:
            return

        # For this example, we'll just limit the number of keyframes
        # to prevent memory growth
        if len(self.keyframes) > 50:  # Keep only recent keyframes
            self.keyframes = self.keyframes[-30:]  # Keep last 30

    def detect_loop_closure(self):
        """Detect potential loop closures."""
        if len(self.keyframes) < 5:
            return

        # Get the most recent keyframe
        recent_kf = self.keyframes[-1]

        # Check against previous keyframes
        for kf in self.keyframes[:-5]:  # Skip last 5 to avoid immediate matches
            if self.loop_detector.is_similar(recent_kf, kf):
                print(f"Potential loop closure detected between {recent_kf.id} and {kf.id}")
                # In a real system, this would trigger pose graph optimization
                break

class LoopClosureDetector:
    """Detects when the robot revisits a location."""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.bow_vocabulary = None  # Would use BoW vocabulary in practice

    def is_similar(self, kf1: Keyframe, kf2: Keyframe) -> bool:
        """Check if two keyframes represent the same location."""
        # Simple implementation using histogram comparison
        hist1 = self.compute_histogram(kf1.image)
        hist2 = self.compute_histogram(kf2.image)

        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        return correlation > self.similarity_threshold

    def compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """Compute image histogram for similarity comparison."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

# Example usage for humanoid robot
def main():
    # Camera parameters for humanoid robot (example values)
    camera_matrix = np.array([
        [615.0, 0.0, 320.0],  # fx, 0, cx
        [0.0, 615.0, 240.0],  # 0, fy, cy
        [0.0, 0.0, 1.0]       # 0, 0, 1
    ])

    distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # No distortion

    # Initialize SLAM system
    slam = VisualSLAM(camera_matrix, distortion_coeffs)

    # Simulate processing a video stream
    print("Visual SLAM initialized for humanoid robot")
    print("Processing frames...")

    # In a real application, you would read from a camera
    # For this example, we'll just simulate the process
    for i in range(100):
        # Simulate getting an image (in real app, this would come from camera)
        # Create a synthetic image for demonstration
        synthetic_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Process the frame
        current_pose = slam.process_frame(synthetic_image)

        if i % 10 == 0:  # Print every 10 frames
            print(f"Frame {i}: Position = {current_pose.position}")

    print("SLAM processing complete")
    print(f"Created {len(slam.keyframes)} keyframes")
    print(f"Estimated final position: {slam.current_pose.position}")

if __name__ == "__main__":
    main()
```

### 2. Humanoid-Specific Localization

For humanoid robots, we need specialized localization that accounts for their unique characteristics:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d

class HumanoidLocalization:
    """Enhanced localization system for humanoid robots."""

    def __init__(self,
                 camera_height: float = 1.5,  # Humanoid camera height
                 step_length: float = 0.7,     # Average step length
                 max_height_change: float = 0.5):  # Max height change for relocalization

        self.camera_height = camera_height
        self.step_length = step_length
        self.max_height_change = max_height_change

        # Initialize particle filter for global localization
        self.particle_filter = ParticleFilter()

        # For tracking walking motion
        self.prev_foot_pos = None
        self.step_count = 0

        # For handling humanoid-specific movements
        self.movement_state = "walking"  # walking, standing, sitting, climbing
        self.height_offset = 0.0  # For sitting/standing height changes

    def update_from_vslam(self, vslam_pose: Pose, vslam_confidence: float) -> Pose:
        """Update localization using VSLAM estimate with humanoid constraints."""

        # Apply humanoid-specific corrections
        corrected_pose = self.apply_humanoid_constraints(vslam_pose)

        # Fuse with particle filter for global consistency
        self.particle_filter.update_measurement(corrected_pose, vslam_confidence)

        # Get the best estimate
        best_pose = self.particle_filter.get_best_estimate()

        return best_pose

    def apply_humanoid_constraints(self, vslam_pose: Pose) -> Pose:
        """Apply humanoid-specific constraints to VSLAM pose."""

        # Adjust for camera height to get foot position
        adjusted_pose = Pose(
            position=np.array([
                vslam_pose.position[0],
                vslam_pose.position[1],
                vslam_pose.position[2] - self.camera_height
            ]),
            rotation=vslam_pose.rotation
        )

        # Check for physically impossible movements
        if self.is_movement_valid(adjusted_pose):
            return adjusted_pose
        else:
            # Return previous valid pose or a constrained version
            print("Warning: Physically impossible movement detected")
            return self.handle_invalid_movement(adjusted_pose)

    def is_movement_valid(self, pose: Pose) -> bool:
        """Check if the movement is physically valid for a humanoid."""
        if not hasattr(self, 'prev_pose') or self.prev_pose is None:
            self.prev_pose = pose
            return True

        # Calculate movement
        movement = pose.position - self.prev_pose.position
        movement_2d = np.linalg.norm(movement[:2])  # Horizontal movement
        height_change = abs(movement[2])  # Vertical movement

        # Check if movement is too large (impossible for humanoid)
        if movement_2d > self.step_length * 2:  # More than 2 steps at once
            return False

        # Check if vertical movement is too large
        if height_change > self.max_height_change:
            return False

        return True

    def handle_invalid_movement(self, pose: Pose) -> Pose:
        """Handle invalid movements in humanoid-specific way."""
        if hasattr(self, 'prev_pose') and self.prev_pose is not None:
            # Return previous pose as fallback
            return self.prev_pose
        else:
            # If no previous pose, return the input pose
            return pose

    def detect_locomotion_mode(self, pose_change: Pose, sensor_data: Dict) -> str:
        """Detect current locomotion mode based on pose and sensor data."""

        # Use accelerometer and gyroscope data to detect movement pattern
        linear_accel = sensor_data.get('linear_accel', np.array([0, 0, 0]))
        angular_vel = sensor_data.get('angular_vel', np.array([0, 0, 0]))

        # Calculate movement characteristics
        linear_accel_mag = np.linalg.norm(linear_accel)
        angular_vel_mag = np.linalg.norm(angular_vel)

        # Detect walking based on periodic patterns
        if linear_accel_mag > 0.5 and angular_vel_mag > 0.1:
            # Check for walking periodicity (simplified)
            if hasattr(self, 'walking_pattern') and self.walking_pattern:
                return "walking"
            else:
                return "moving"
        elif linear_accel_mag < 0.2:
            return "standing"
        else:
            return "other"

    def update_height_for_posture(self, posture: str):
        """Update camera height based on humanoid posture."""
        if posture == "standing":
            self.camera_height = 1.5
            self.height_offset = 0.0
        elif posture == "sitting":
            self.camera_height = 0.8
            self.height_offset = 0.7
        elif posture == "crouching":
            self.camera_height = 0.6
            self.height_offset = 0.9
        else:
            # Default to standing height
            self.camera_height = 1.5
            self.height_offset = 0.0

class ParticleFilter:
    """Simple particle filter for global localization."""

    def __init__(self, num_particles: int = 1000):
        self.num_particles = num_particles
        self.particles = self.initialize_particles()
        self.weights = np.ones(num_particles) / num_particles

    def initialize_particles(self) -> np.ndarray:
        """Initialize particles randomly in the environment."""
        # For this example, initialize in a 10x10m area
        particles = np.random.uniform(-5, 5, (self.num_particles, 3))  # x, y, z
        # Add orientation (for simplicity, just z-rotation)
        angles = np.random.uniform(-np.pi, np.pi, (self.num_particles, 1))
        return np.hstack([particles, angles])

    def update_measurement(self, measurement: Pose, confidence: float):
        """Update particle weights based on measurement."""
        # Convert measurement to particle format [x, y, z, theta]
        meas_vec = np.array([
            measurement.position[0],
            measurement.position[1],
            measurement.position[2],
            self.quaternion_to_yaw(measurement.rotation)
        ])

        # Calculate distances and update weights
        for i in range(self.num_particles):
            particle_pose = self.particles[i, :4]  # x, y, z, theta
            dist = np.linalg.norm(meas_vec - particle_pose)

            # Weight based on distance and confidence
            weight_update = confidence * np.exp(-dist * 0.1)  # Simplified model
            self.weights[i] *= weight_update + 0.01  # Add small constant to prevent 0

        # Normalize weights
        self.weights /= np.sum(self.weights)

        # Resample if effective number of particles is too low
        if self.effective_particles() < self.num_particles / 2:
            self.resample()

    def effective_particles(self) -> float:
        """Calculate effective number of particles."""
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self):
        """Resample particles based on weights."""
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )

        self.particles = self.particles[indices, :]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_best_estimate(self) -> Pose:
        """Get the best pose estimate from particles."""
        # Weighted average of particles
        avg_pos = np.average(self.particles[:, :3], axis=0, weights=self.weights)
        avg_yaw = np.arctan2(
            np.average(np.sin(self.particles[:, 3]), weights=self.weights),
            np.average(np.cos(self.particles[:, 3]), weights=self.weights)
        )

        # Convert yaw to quaternion
        quat = self.yaw_to_quaternion(avg_yaw)

        return Pose(position=avg_pos, rotation=quat)

    def quaternion_to_yaw(self, quat: np.ndarray) -> float:
        """Extract yaw angle from quaternion."""
        # Simplified: extract yaw from quaternion
        siny_cosp = 2 * (quat[3] * quat[2] + quat[0] * quat[1])
        cosy_cosp = 1 - 2 * (quat[1]**2 + quat[2]**2)
        return np.arctan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw: float) -> np.ndarray:
        """Convert yaw angle to quaternion."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return np.array([0, 0, sy, cy])

# Example integration with humanoid robot system
class HumanoidNavigationSystem:
    """Complete navigation system for humanoid robots."""

    def __init__(self):
        # Initialize VSLAM system
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.vslam = VisualSLAM(self.camera_matrix, np.zeros(5))

        # Initialize humanoid-specific localization
        self.localization = HumanoidLocalization()

        # For path planning
        self.global_path = []
        self.local_path = []

    def process_sensor_data(self,
                           image: np.ndarray,
                           imu_data: Dict,
                           encoder_data: Dict) -> Dict:
        """Process all sensor data for navigation."""

        # Process VSLAM
        vslam_pose = self.vslam.process_frame(image)

        # Get IMU-based motion estimate for validation
        imu_estimate = self.process_imu_data(imu_data)

        # Combine estimates
        combined_pose = self.fuse_sensors(vslam_pose, imu_estimate)

        # Update humanoid-specific localization
        final_pose = self.localization.update_from_vslam(
            combined_pose,
            vslam_confidence=0.8
        )

        # Update locomotion mode
        self.localization.movement_state = self.localization.detect_locomotion_mode(
            combined_pose,
            imu_data
        )

        return {
            'pose': final_pose,
            'state': self.localization.movement_state,
            'path': self.global_path
        }

    def process_imu_data(self, imu_data: Dict) -> Pose:
        """Process IMU data for motion estimation."""
        # Simplified IMU integration
        linear_accel = np.array(imu_data.get('linear_accel', [0, 0, 0]))
        angular_vel = np.array(imu_data.get('angular_vel', [0, 0, 0]))

        # Integrate to get pose change (simplified)
        dt = 0.01  # 100Hz assumption
        linear_vel = linear_accel * dt
        angular_pos = angular_vel * dt

        # Convert to pose change
        position_change = linear_vel * dt
        rotation_change = angular_pos

        # Create pose change
        pose_change = Pose(
            position=position_change,
            rotation=self.euler_to_quaternion(rotation_change)
        )

        return pose_change

    def fuse_sensors(self, vslam_pose: Pose, imu_pose: Pose) -> Pose:
        """Fuse VSLAM and IMU estimates."""
        # Simple weighted fusion (in practice, use EKF or UKF)
        weight_vslam = 0.7
        weight_imu = 0.3

        fused_position = (weight_vslam * vslam_pose.position +
                         weight_imu * imu_pose.position)
        fused_rotation = self.slerp_quaternions(
            vslam_pose.rotation,
            imu_pose.rotation,
            weight_imu
        )

        return Pose(position=fused_position, rotation=fused_rotation)

    def euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to quaternion."""
        roll, pitch, yaw = euler

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([x, y, z, w])

    def slerp_quaternions(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions."""
        # Calculate cosine of angle
        cos_half_theta = np.dot(q1, q2)

        if cos_half_theta < 0:
            q2 = -q2
            cos_half_theta = -cos_half_theta

        if cos_half_theta >= 1.0:
            return q1

        half_theta = np.arccos(cos_half_theta)
        sin_half_theta = np.sqrt(1 - cos_half_theta**2)

        if abs(sin_half_theta) < 0.001:
            return 0.5 * q1 + 0.5 * q2

        ratio_a = np.sin((1 - t) * half_theta) / sin_half_theta
        ratio_b = np.sin(t * half_theta) / sin_half_theta

        return ratio_a * q1 + ratio_b * q2

# Example usage
def run_humanoid_navigation():
    """Example of using the humanoid navigation system."""
    nav_system = HumanoidNavigationSystem()

    # Simulate sensor data
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_imu = {
        'linear_accel': [0.1, 0.0, 9.81],  # Slightly moving forward
        'angular_vel': [0.0, 0.0, 0.1]      # Turning slightly
    }
    dummy_encoders = {'left': 0.1, 'right': 0.1}  # Wheel encoders (simulated)

    # Process the sensor data
    result = nav_system.process_sensor_data(dummy_image, dummy_imu, dummy_encoders)

    print(f"Current pose: {result['pose'].position}")
    print(f"Current state: {result['state']}")

    return result

if __name__ == "__main__":
    result = run_humanoid_navigation()
```

### 3. Integration with ROS2

Here's how to integrate VSLAM with ROS2 for humanoid robots:

```python
#!/usr/bin/env python3
"""
ROS2 node for Visual SLAM and localization for humanoid robots.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf2_ros import TransformBroadcaster
import tf_transformations as tft

class HumanoidVSLAMNode(Node):
    """ROS2 node for humanoid Visual SLAM and localization."""

    def __init__(self):
        super().__init__('humanoid_vslam_node')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize VSLAM system
        self.initialize_slam_system()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Create publishers
        self.odom_pub = self.create_publisher(Odometry, '/vslam/odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/vslam/pose', 10)

        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Parameters
        self.declare_parameter('camera_height', 1.5)
        self.declare_parameter('step_length', 0.7)

        self.camera_height = self.get_parameter('camera_height').value
        self.step_length = self.get_parameter('step_length').value

        # State variables
        self.camera_matrix = None
        self.prev_image = None
        self.prev_time = None

        self.get_logger().info('Humanoid VSLAM node initialized')

    def initialize_slam_system(self):
        """Initialize the SLAM system."""
        # Default camera matrix (will be updated from camera_info)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

        self.distortion_coeffs = np.zeros(5)

        # Initialize SLAM
        self.vslam = VisualSLAM(self.camera_matrix, self.distortion_coeffs)
        self.localization = HumanoidLocalization(self.camera_height, self.step_length)

    def image_callback(self, msg):
        """Process incoming camera images."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Get current time
            current_time = self.get_clock().now()

            # Process with VSLAM
            pose = self.vslam.process_frame(cv_image)

            # Create Odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = current_time.to_msg()
            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = 'base_link'

            # Set position
            odom_msg.pose.pose.position.x = float(pose.position[0])
            odom_msg.pose.pose.position.y = float(pose.position[1])
            odom_msg.pose.pose.position.z = float(pose.position[2])

            # Set orientation
            odom_msg.pose.pose.orientation.x = float(pose.rotation[0])
            odom_msg.pose.pose.orientation.y = float(pose.rotation[1])
            odom_msg.pose.pose.orientation.z = float(pose.rotation[2])
            odom_msg.pose.pose.orientation.w = float(pose.rotation[3])

            # Publish odometry
            self.odom_pub.publish(odom_msg)

            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = current_time.to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose = odom_msg.pose.pose

            self.pose_pub.publish(pose_msg)

            # Broadcast transform
            self.broadcast_transform(current_time, pose)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def imu_callback(self, msg):
        """Process IMU data."""
        # Store IMU data for fusion with VSLAM
        self.imu_data = {
            'linear_accel': np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ]),
            'angular_vel': np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
        }

    def camera_info_callback(self, msg):
        """Update camera parameters from camera info."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)

            # Update SLAM system with actual camera parameters
            self.vslam.camera_matrix = self.camera_matrix
            self.vslam.distortion_coeffs = self.distortion_coeffs

    def broadcast_transform(self, timestamp, pose):
        """Broadcast the robot's transform."""
        from geometry_msgs.msg import TransformStamped

        t = TransformStamped()

        t.header.stamp = timestamp.to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = pose.position[0]
        t.transform.translation.y = pose.position[1]
        t.transform.translation.z = pose.position[2]

        t.transform.rotation.x = pose.rotation[0]
        t.transform.rotation.y = pose.rotation[1]
        t.transform.rotation.z = pose.rotation[2]
        t.transform.rotation.w = pose.rotation[3]

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)

    vslam_node = HumanoidVSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```mermaid
graph TD
    A[Camera Input] --> B[Feature Extraction]
    B --> C[Visual Odometry]
    C --> D[Pose Estimation]
    D --> E[Map Building]
    E --> F[Loop Closure]
    F --> G[Graph Optimization]
    G --> H[Final Pose]
    H --> I[Humanoid Navigation]

    J[IMU Data] --> K[Sensor Fusion]
    L[Encoders] --> K
    K --> D

    I --> M[Path Planning]
    M --> N[Obstacle Avoidance]
    N --> O[Humanoid Control]

    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style G fill:#e8f5e8
    style I fill:#fff3e0
    style O fill:#e0f2f1
</graph>

## Troubleshooting

### Common Issues and Solutions

#### 1. Drift and Accumulated Error
**Symptoms**: Robot's estimated position diverges significantly from actual position over time
**Solutions**:
- Implement robust loop closure detection
- Use pose graph optimization regularly
- Fuse with other sensors (IMU, encoders)
- Improve feature tracking stability
- Add motion model constraints

#### 2. Tracking Failure
**Symptoms**: SLAM system loses track and cannot recover
**Solutions**:
- Implement relocalization mechanisms
- Use multiple feature detectors as fallbacks
- Add visual-inertial fusion for better tracking
- Improve initialization procedures
- Handle rapid motion better

#### 3. Scale Drift (Monocular SLAM)
**Symptoms**: Estimated distances change over time in monocular systems
**Solutions**:
- Use stereo or RGB-D when possible
- Add scale constraints from humanoid kinematics
- Use object size priors in the environment
- Implement online scale estimation

#### 4. Computational Performance
**Symptoms**: System cannot run in real-time on humanoid robot
**Solutions**:
- Optimize feature detection and matching
- Use keyframe-based approaches to reduce computation
- Implement multi-threading for different SLAM components
- Use hardware acceleration (GPUs)
- Adjust parameters for speed vs. accuracy trade-off

#### 5. Dynamic Objects
**Symptoms**: Moving objects cause tracking errors and wrong map updates
**Solutions**:
- Implement dynamic object detection and removal
- Use optical flow to identify moving pixels
- Apply temporal consistency checks
- Segment and track dynamic objects separately

:::tip
For humanoid robots, consider using multiple SLAM systems running in parallel - one for global mapping and another for local obstacle avoidance, which can handle dynamic objects better.
:::

:::warning
Always validate VSLAM performance in real environments before deploying to humanoid robots, as simulation may not capture all challenges of real-world operation.
:::

:::danger
Never rely solely on VSLAM for safety-critical operations in humanoid robots. Always have fallback navigation methods and emergency stop procedures.
:::

### Performance Optimization

For humanoid robot applications, consider these optimizations:

1. **Adaptive Resolution**: Adjust image resolution based on computational budget
2. **Feature Selection**: Use the most stable and distinctive features
3. **Multi-threading**: Separate tracking and mapping threads
4. **Memory Management**: Implement proper memory pooling to avoid allocation overhead
5. **Hardware Acceleration**: Use GPUs for feature detection and matching

## Summary

This chapter covered Visual SLAM and localization techniques specifically for humanoid robotics:

1. **Fundamentals**: Understanding VSLAM principles and approaches (monocular, stereo, RGB-D)
2. **Humanoid Challenges**: Addressing unique challenges like bipedal motion and height variations
3. **Implementation**: Complete VSLAM system with humanoid-specific localization
4. **Integration**: Connecting VSLAM with ROS2 for humanoid robot applications
5. **Optimization**: Techniques for real-time performance on humanoid platforms
6. **Troubleshooting**: Common issues and solutions in VSLAM systems

Visual SLAM provides humanoid robots with the ability to understand and navigate in unknown environments using only visual sensors. The key to success lies in properly addressing humanoid-specific challenges and ensuring robust performance in dynamic environments.

The integration of VSLAM with other sensors and control systems is crucial for creating truly autonomous humanoid robots capable of operating in human environments.

## Further Reading

1. [Visual SLAM for Humanoid Robots: A Survey](https://ieeexplore.ieee.org/document/8956432) - Comprehensive survey of VSLAM techniques for humanoid applications

2. [ORB-SLAM: A Versatile and Accurate Monocular SLAM System](https://arxiv.org/abs/1502.00956) - Foundational paper on feature-based SLAM

3. [LSD-SLAM: Large-Scale Direct Monocular SLAM](https://arxiv.org/abs/1402.0022) - Direct method for large-scale environments

4. [RGB-D Bundle Adjustment for Dense Reconstruction of Indoor Scenes](https://ieeexplore.ieee.org/document/7410712) - RGB-D SLAM techniques

5. [Humanoid Robot Navigation: A Survey](https://www.sciencedirect.com/science/article/pii/S0921889021000452) - Survey on navigation approaches for humanoid robots