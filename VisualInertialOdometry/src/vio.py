import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from gtsam import (
    ISAM2, NonlinearFactorGraph, Values, Pose3, Point3, Rot3,
    PreintegrationParams, PreintegratedImuMeasurements,
    PriorFactorPose3, PriorFactorVector, PriorFactorConstantBias,
    BetweenFactorConstantBias, ImuFactor, noiseModel, imuBias, BetweenFactorPose3, ISAM2Params)
import gtsam
from visualization import plotting
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from gtsam.symbol_shorthand import X, V, B
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pandas as pd
import csv
import glob, math, json

# Import your MoCap helper
from plot_mocap_xy import MocapHelper

############################################# FEATURE TYPE ############################################
FEATURE_TYPE = "SIFT"  # SIFT, or FAST
#######################################################################################################

# -------------------------
# YOUR ABSOLUTE DATA PATHS
# -------------------------
CALIB_PATH   = r"C:\VIOCODE\ComputerVision\VisualOdometry\calib.txt"
MOCAP_CSV    = Path(r"C:\VIOCODE\ComputerVision\VisualOdometry\MOCAP\Husky_curve.csv")
LEFT_DIR     = r"C:\VIOCODE\ComputerVision\VisualOdometry\curv_underwater\left_image"
RIGHT_DIR    = r"C:\VIOCODE\ComputerVision\VisualOdometry\curv_underwater\right_image"
LEFT_DATACSV = r"C:\VIOCODE\ComputerVision\VisualOdometry\curv\left_image\data.csv"
RIGHT_DATACSV= r"C:\VIOCODE\ComputerVision\VisualOdometry\curv\right_image\data.csv"
IMU_CSV      = r"C:\VIOCODE\ComputerVision\VisualOdometry\curv\imu\imu_data.csv"

# Coordinate frame transformations #
def umeyama_alignment(estimated_poses, gt_poses):
    """
    Find optimal transformation (rotation + translation) that aligns estimated trajectory to GT
    using Umeyama's method (point cloud registration)
    """
    # Extract positions
    est_positions = np.array([pose[:3, 3] for pose in estimated_poses])
    gt_positions = np.array([pose[:3, 3] for pose in gt_poses])
    
    # Center the point clouds
    est_centered = est_positions - np.mean(est_positions, axis=0)
    gt_centered = gt_positions - np.mean(gt_positions, axis=0)
    
    # Compute covariance matrix
    H = est_centered.T @ gt_centered
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Calculate rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = np.mean(gt_positions, axis=0) - R @ np.mean(est_positions, axis=0)
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def align_trajectory_umeyama(estimated_poses, gt_poses):
    """Align entire estimated trajectory to GT using Umeyama"""
    T_align = umeyama_alignment(estimated_poses, gt_poses)
    
    aligned_poses = []
    for pose in estimated_poses:
        aligned_pose = T_align @ pose
        aligned_poses.append(aligned_pose)
    
    return aligned_poses
def plot_relative_trajectory(estimated_poses, gt_poses):
    """Plot trajectories relative to their starting points"""
    
    # Extract positions relative to first pose
    est_positions = np.array([pose[:3, 3] for pose in estimated_poses])
    gt_positions = np.array([pose[:3, 3] for pose in gt_poses])
    
    # Make relative to start
    est_relative = est_positions - est_positions[0]
    gt_relative = gt_positions - gt_positions[0]
    
    # Simple 2D plot (ignore height)
    plt.figure(figsize=(12, 10))
    plt.subplot(1, 2, 1)
    plt.plot(-est_relative[:, 1], -est_relative[:, 0], 'b-', label='Estimated', linewidth=2)
    plt.plot(gt_relative[:, 0], gt_relative[:, 2], 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Relative Trajectory (Top View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # plt.subplot(1, 2, 2)
    # plt.plot(est_relative[:, 0], est_relative[:, 2], 'b-', label='Estimated', linewidth=2)
    # plt.plot(gt_relative[:, 0], gt_relative[:, 2], 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
    # plt.xlabel('X [m]')
    # plt.ylabel('Z [m]')
    # plt.title('Relative Trajectory (Side View)')
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return -est_relative, gt_relative
def euler_to_rotation_matrix(rx, ry, rz):
    """Convert Euler angles (in radians) to rotation matrix.
    Assumes rotation order: ZYX (yaw, pitch, roll)
    """
    # Create rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: ZYX order
    R = Rz @ Ry @ Rx
    return R

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (ZYX order)"""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        rx = np.arctan2(R[2,1], R[2,2])  # roll
        ry = np.arctan2(-R[2,0], sy)      # pitch  
        rz = np.arctan2(R[1,0], R[0,0])   # yaw
    else:
        rx = np.arctan2(-R[1,2], R[1,1])
        ry = np.arctan2(-R[2,0], sy)
        rz = 0
        
    return rx, ry, rz

def determine_world_to_body_alignment(initial_gt_pose, initial_imu_orientation=None, mocap_to_body_rotation=None):
    """
    EULER-PRESERVING: Identity transformation for orientation
    """
    print("=== EULER-PRESERVING WORLD TO BODY ALIGNMENT ===")
    
    # Use identity transformation - preserves Euler angle interpretations
    R_world_body = np.eye(3)
    
    print("Using identity transformation for orientation")
    print("Euler angles will be interpreted consistently between MoCap and Body frames")
    
    return R_world_body

def verify_coordinate_transformation(vo, gt_poses_world):
    """Verify that the coordinate transformation makes sense"""
    print("\n=== COORDINATE TRANSFORMATION VERIFICATION ===")
    
    # Test the first pose transformation
    if not gt_poses_world:
        return
    
    initial_pose_world = gt_poses_world[0]
    initial_pose_body = vo.world_to_body_pose(initial_pose_world)
    
    # Extract positions and rotations
    pos_world = initial_pose_world[:3, 3]
    pos_body = initial_pose_body[:3, 3]
    
    rot_world = initial_pose_world[:3, :3]
    rot_body = initial_pose_body[:3, :3]
    
    rx_w, ry_w, rz_w = rotation_matrix_to_euler(rot_world)
    rx_b, ry_b, rz_b = rotation_matrix_to_euler(rot_body)
    
    print("First pose transformation:")
    print(f"  MoCap Frame:")
    print(f"    Position: X(left/right)={pos_world[0]:.3f}, Y(up/down)={pos_world[1]:.3f}, Z(forward/back)={pos_world[2]:.3f}")
    print(f"    Rotation: Roll={np.degrees(rx_w):.1f}°, Pitch={np.degrees(ry_w):.1f}°, Yaw={np.degrees(rz_w):.1f}°")
    
    print(f"  Body Frame:")
    print(f"    Position: X(forward)={pos_body[0]:.3f}, Y(left/right)={pos_body[1]:.3f}, Z(up/down)={pos_body[2]:.3f}")
    print(f"    Rotation: Roll={np.degrees(rx_b):.1f}°, Pitch={np.degrees(ry_b):.1f}°, Yaw={np.degrees(rz_b):.1f}°")
    
    # The first pose in body frame should be near origin
    dist_from_origin = np.linalg.norm(pos_body)
    print(f"  Distance from origin in body frame: {dist_from_origin:.3f}m (should be near 0)")
    
    # Test a few more poses to see the transformation makes sense
    print("\nTesting transformation consistency on first 3 poses:")
    for i in range(min(3, len(gt_poses_world))):
        pose_world = gt_poses_world[i]
        pose_body = vo.world_to_body_pose(pose_world)
        
        pos_w = pose_world[:3, 3]
        pos_b = pose_body[:3, 3]
        
        # The Z coordinate in MoCap (forward) should become X in Body (forward)
        # The X coordinate in MoCap (left/right) should become Y in Body (left/right)
        print(f"  Pose {i}: MoCap(Z,X)=({pos_w[2]:.2f}, {pos_w[0]:.2f}) -> Body(X,Y)=({pos_b[0]:.2f}, {pos_b[1]:.2f})")

def apply_world_to_body_transform(pose_world, R_world_body, t_world_body=None):
    """
    Transform only position, keep orientation as-is
    """
    if t_world_body is None:
        t_world_body = np.zeros(3)
    
    R_world = pose_world[:3, :3]
    t_world = pose_world[:3, 3]
    
    # Transform position using axis swapping
    t_body = np.array([
        t_world[2] - t_world_body[2],  # MoCap Z -> Body X (forward)
        t_world[0] - t_world_body[0],  # MoCap X -> Body Y (left/right)
        t_world[1] - t_world_body[1]   # MoCap Y -> Body Z (up/down)
    ])
    
    # Keep orientation matrix as-is (no rotation transformation)
    R_body = R_world
    
    pose_body = np.eye(4)
    pose_body[:3, :3] = R_body
    pose_body[:3, 3] = t_body
    
    return pose_body
def evaluate_using_relative_motion(estimated_poses, gt_poses):
    """Evaluate using relative motion between consecutive poses"""
    eval_len = min(len(estimated_poses), len(gt_poses))
    
    if eval_len < 2:
        print("Need at least 2 poses for relative motion evaluation")
        return None
    
    position_errors = []
    orientation_errors = []
    relative_distances = []
    
    for i in range(1, eval_len):
        # Calculate relative transforms (movement from frame i-1 to i)
        est_rel = np.linalg.inv(estimated_poses[i-1]) @ estimated_poses[i]
        gt_rel = np.linalg.inv(gt_poses[i-1]) @ gt_poses[i]
        
        # Relative position error
        est_rel_pos = est_rel[:3, 3]
        gt_rel_pos = gt_rel[:3, 3]
        pos_error = np.linalg.norm(est_rel_pos - gt_rel_pos)
        position_errors.append(pos_error)
        
        # Ground truth movement distance (for context)
        gt_distance = np.linalg.norm(gt_rel_pos)
        relative_distances.append(gt_distance)
        
        # Relative orientation error
        est_rel_rot = est_rel[:3, :3]
        gt_rel_rot = gt_rel[:3, :3]
        R_diff = est_rel_rot.T @ gt_rel_rot
        orient_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        orientation_errors.append(orient_error)
    
    avg_pos_error = np.mean(position_errors)
    avg_orient_error = np.degrees(np.mean(orientation_errors))
    avg_gt_distance = np.mean(relative_distances)
    
    print(f"Relative Motion Evaluation ({len(position_errors)} steps):")
    print(f"  Average position error: {avg_pos_error:.3f} m/step")
    print(f"  Average orientation error: {avg_orient_error:.1f}°/step") 
    print(f"  Average ground truth movement: {avg_gt_distance:.3f} m/step")
    print(f"  Position error as % of movement: {(avg_pos_error/avg_gt_distance)*100:.1f}%")
    
    return {
        'position_errors': position_errors,
        'orientation_errors': orientation_errors,
        'relative_distances': relative_distances,
        'avg_position_error': avg_pos_error,
        'avg_orientation_error': avg_orient_error
    }

def apply_body_to_world_transform(pose_body, R_world_body, t_world_body=None):
    """
    CORRECT: Transform a pose from body frame to world frame
    """
    if t_world_body is None:
        t_world_body = np.zeros(3)
    
    R_body = pose_body[:3, :3]
    t_body = pose_body[:3, 3]
    
    # CORRECT INVERSE TRANSFORMATION:
    R_world = R_world_body.T @ R_body  # Body rotation to world rotation
    t_world = R_world_body.T @ t_body + t_world_body  # Body position to world position
    
    pose_world = np.eye(4)
    pose_world[:3, :3] = R_world
    pose_world[:3, 3] = t_world
    
    return pose_world

# END OF COORDINATE FRAME TRANSFORMATIONS
@dataclass
class IMUSample:
    timestamp: float
    accel: np.ndarray
    gyro: np.ndarray
    quat_wxyz: Optional[np.ndarray] = None

class FeatureBackend:
    def __init__(self):
        pass
    
    def detect_and_describe(self, img_gray):
        raise NotImplementedError
        
    def match_features(self, img1, img2, pts1, des1):
        """Unified feature matching interface"""
        raise NotImplementedError

class ORBBackend(FeatureBackend):
    def __init__(self, nfeatures=2000, ratio=0.75):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        self.ratio = ratio
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_describe(self, img_gray):
        kps, des = self.orb.detectAndCompute(img_gray, None)
        if des is None or len(kps) == 0:
            return np.empty((0, 2), dtype=np.float32), None
        pts = np.asarray([k.pt for k in kps], dtype=np.float32)
        return pts, des

    def match_features(self, img1, img2, pts1, des1):
        if des1 is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
            
        # Detect features in second image
        pts2, des2 = self.detect_and_describe(img2)
        if len(pts2) == 0 or des2 is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
        # Match features
        knn = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < self.ratio * n.distance:
                good.append(m)
        
        if len(good) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
        # Get matched points
        idx1 = np.array([m.queryIdx for m in good], dtype=np.int32)
        idx2 = np.array([m.trainIdx for m in good], dtype=np.int32)
        
        matched_pts1 = pts1[idx1]
        matched_pts2 = pts2[idx2]
        
        return matched_pts1, matched_pts2

class SIFTBackend(FeatureBackend):
    def __init__(self, nfeatures=1000, contrastThreshold=0.03, nOctaveLayers=4, ratio=0.7):
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,           # Limit total features
            contrastThreshold=0.03,        # Medium - good balance
            nOctaveLayers=4,               # Better scale invariance
            edgeThreshold=15,              # Higher - filter edge noise
            sigma=1.6                      # Default
        )
        self.ratio = ratio
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    def detect_and_describe(self, img_gray):
        kps, des = self.sift.detectAndCompute(img_gray, None)
        if des is None or len(kps) == 0:
            return np.empty((0, 2), dtype=np.float32), None
        pts = np.asarray([k.pt for k in kps], dtype=np.float32)
        return pts, des

    def match_features(self, img1, img2, pts1, des1):
        if des1 is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)  
        # Detect features in second image
        pts2, des2 = self.detect_and_describe(img2)
        if len(pts2) == 0 or des2 is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
        # Match features using FLANN
        knn = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < self.ratio * n.distance:
                good.append(m)
        
        if len(good) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
        # Get matched points
        idx1 = np.array([m.queryIdx for m in good], dtype=np.int32)
        idx2 = np.array([m.trainIdx for m in good], dtype=np.int32)
        
        matched_pts1 = pts1[idx1]
        matched_pts2 = pts2[idx2]
        
        return matched_pts1, matched_pts2

class FASTBackend(FeatureBackend):
    def __init__(self, threshold=20, nonmaxSuppression=True, max_error=3.0):  # Reduced max_error
        self.fast = cv2.FastFeatureDetector_create(threshold=threshold, 
                                                  nonmaxSuppression=nonmaxSuppression)
        # Improved LK parameters for better tracking
        self.lk_params = dict(
            winSize=(21, 21),  # Larger window for better tracking
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)  # Stricter criteria
        )
        self.max_error = max_error

    def detect_and_describe(self, img_gray):
        # Add some preprocessing to improve feature detection
        if img_gray is None:
            return np.empty((0, 2), dtype=np.float32), None
            
        # Apply histogram equalization to improve contrast
        img_eq = cv2.equalizeHist(img_gray)
        
        kps = self.fast.detect(img_eq, None)
        if len(kps) == 0:
            return np.empty((0, 2), dtype=np.float32), None
        pts = np.asarray([k.pt for k in kps], dtype=np.float32)
        return pts, None

    def match_features(self, img1, img2, pts1, des1):
        if len(pts1) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
        # Preprocess images for better optical flow
        img1_processed = cv2.equalizeHist(img1) if img1 is not None else img1
        img2_processed = cv2.equalizeHist(img2) if img2 is not None else img2
        
        # Use optical flow for FAST features
        trackpoints1 = np.expand_dims(pts1, axis=1).astype(np.float32)
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(
            img1_processed, img2_processed, trackpoints1, None, **self.lk_params
        )

        if trackpoints2 is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        trackable = st.astype(bool).flatten()
        
        if np.sum(trackable) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        
        # Get the trackable points and errors
        trackpoints1_trackable = trackpoints1[trackable]
        trackpoints2_trackable = trackpoints2[trackable]
        err_trackable = err[trackable]
        
        # Filter by error threshold (more strict)
        under_thresh = err_trackable.flatten() < self.max_error
        trackpoints1_filtered = trackpoints1_trackable[under_thresh]
        trackpoints2_filtered = trackpoints2_trackable[under_thresh]
        
        if len(trackpoints1_filtered) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        # Reshape to (N, 2) and filter by image bounds
        h, w = img1.shape
        trackpoints1_flat = trackpoints1_filtered.reshape(-1, 2)
        trackpoints2_flat = trackpoints2_filtered.reshape(-1, 2)
        
        # Check bounds with some margin
        margin = 5
        in_bounds = np.logical_and(
            np.logical_and(trackpoints2_flat[:, 0] >= margin, trackpoints2_flat[:, 0] < w - margin),
            np.logical_and(trackpoints2_flat[:, 1] >= margin, trackpoints2_flat[:, 1] < h - margin)
        )
        
        matched_pts1 = trackpoints1_flat[in_bounds]
        matched_pts2 = trackpoints2_flat[in_bounds]
        
        # Additional filtering: remove points that moved too little (noise)
        if len(matched_pts1) > 0:
            movements = np.linalg.norm(matched_pts2 - matched_pts1, axis=1)
            min_movement = 0.5  # pixels
            sufficient_movement = movements > min_movement
            matched_pts1 = matched_pts1[sufficient_movement]
            matched_pts2 = matched_pts2[sufficient_movement]
        
        return matched_pts1, matched_pts2
class DataLoader:
    def __init__(self):
        self.mocap_helper = MocapHelper()
    
    def load_mocap_data(self, mocap_path):
        """Load MoCap data as world frame poses with rotation"""
        try:
            print(f"Loading MoCap data from: {mocap_path}")
            df = self.mocap_helper._load_xy_from_motive_csv(Path(mocap_path), body_name="HUSKY")
            print(f"Loaded MoCap data: {len(df)} samples")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Create world frame poses (MoCap frame: x-right, z-forward, y-up)
            poses = []
            for _, row in df.iterrows():
                T = np.eye(4)
                
                # Set position
                T[0, 3] = row['x']  # x position (right/left in world frame)
                T[1, 3] = row['y']  # y position (up/down in world frame)  
                T[2, 3] = row['z']  # z position (forward/back in world frame)
                
                # Set rotation from Euler angles (convert to radians if needed)
                # Assuming angles are in degrees in the CSV
                rx = np.radians(row['rx']) if 'rx' in row else 0.0
                ry = np.radians(row['ry']) if 'ry' in row else 0.0
                rz = np.radians(row['rz']) if 'rz' in row else 0.0
                
                R = euler_to_rotation_matrix(rx, ry, rz)
                T[:3, :3] = R
                
                poses.append(T)
            
            times = df['t'].values
            
            print(f"MoCap time range: {times[0]:.2f} to {times[-1]:.2f}s")
            print(f"MoCap position range: X({np.min([p[0,3] for p in poses]):.2f} to {np.max([p[0,3] for p in poses]):.2f}m), "
                f"Z({np.min([p[2,3] for p in poses]):.2f} to {np.max([p[2,3] for p in poses]):.2f}m)")
            
            # Print rotation info if available
            if 'rx' in df.columns:
                print(f"MoCap rotation range: RX({np.min(df['rx']):.1f} to {np.max(df['rx']):.1f}°), "
                    f"RY({np.min(df['ry']):.1f} to {np.max(df['ry']):.1f}°), "
                    f"RZ({np.min(df['rz']):.1f} to {np.max(df['rz']):.1f}°)")
            
            return times, poses, df
            
        except Exception as e:
            print(f"Error loading MoCap data: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), [], None

    def load_imu_data(self, imu_csv):
        """Load IMU data in body frame and convert to relative timestamps"""
        samples = []
        try:
            with open(imu_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                first_timestamp = None
                for row in reader:
                    try:
                        # Convert nanoseconds to seconds and get absolute timestamp
                        absolute_timestamp = float(row['#timestamp [ns]']) * 1e-9
                        
                        # Store first timestamp to make relative
                        if first_timestamp is None:
                            first_timestamp = absolute_timestamp
                            print(f"First IMU absolute timestamp: {absolute_timestamp:.2f}")
                        
                        # Convert to relative time starting from 0
                        relative_timestamp = absolute_timestamp - first_timestamp
                        
                        accel = np.array([
                            float(row['linear_acceleration_x']),
                            float(row['linear_acceleration_y']), 
                            float(row['linear_acceleration_z'])
                        ])
                        gyro = np.array([
                            float(row['angular_velocity_x']),
                            float(row['angular_velocity_y']),
                            float(row['angular_velocity_z'])
                        ])
                        samples.append(IMUSample(timestamp=relative_timestamp, accel=accel, gyro=gyro))
                    except (KeyError, ValueError):
                        continue
            print(f"Loaded {len(samples)} IMU samples")
            if samples:
                print(f"IMU time range (relative): {samples[0].timestamp:.2f} to {samples[-1].timestamp:.2f}s")
            return samples
        except Exception as e:
            print(f"Error loading IMU data: {e}")
            return []

    def load_camera_data(self, data_csv, left_dir, right_dir):
        """Load camera data in body frame and convert to relative timestamps"""
        # Load timestamps
        times = []
        try:
            with open(data_csv, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                first_timestamp = None
                for row in reader:
                    if len(row) >= 1:
                        try:
                            # Convert nanoseconds to seconds and get absolute timestamp
                            absolute_timestamp = float(row[0]) * 1e-9
                            
                            # Store first timestamp to make relative
                            if first_timestamp is None:
                                first_timestamp = absolute_timestamp
                                print(f"First camera absolute timestamp: {absolute_timestamp:.2f}")
                            
                            # Convert to relative time starting from 0
                            relative_timestamp = absolute_timestamp - first_timestamp
                            times.append(relative_timestamp)
                        except ValueError:
                            continue
            print(f"Loaded {len(times)} camera timestamps")
        except Exception as e:
            print(f"Error loading camera times: {e}")
            times = []
        
        # Load images
        def load_images_from_dir(directory):
            image_files = sorted([f for f in os.listdir(directory) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            images = []
            for f in image_files:
                img = cv2.imread(os.path.join(directory, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
            return images
        
        images_l = load_images_from_dir(left_dir)
        images_r = load_images_from_dir(right_dir)
        
        print(f"Loaded {len(images_l)} left images, {len(images_r)} right images from {left_dir}")
        
        if times:
            print(f"Camera time range (relative): {times[0]:.2f} to {times[-1]:.2f}s")
        
        return times, images_l, images_r

class StereoVO:
    def __init__(self, calib_path, feature_type="ORB"):
        self.calib_path = calib_path
        self.feature_type = feature_type
        
        # Load calibration
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(calib_path)
        
        # Initialize feature backend
        self.backend = self._init_backend(feature_type)
        
        # Stereo matcher
        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2
        )
        self.disparities = []
        
        # GTSAM setup
        self._setup_gtsam()
        
        # Data storage
        self.estimated_poses = []
        self.gt_poses = []
        self.camera_times = []
        
        print(f"Initialized Stereo VO with {feature_type}")
    def align_to_world_frame(self, initial_gt_pose, initial_imu_sample=None):
        """
        Align the VIO system to the world frame using initial GT and IMU data
        """
        print("=== INITIAL WORLD FRAME ALIGNMENT ===")
        
        # Extract initial orientation from IMU if available
        initial_imu_orientation = None
        if initial_imu_sample and initial_imu_sample.quat_wxyz is not None:
            initial_imu_orientation = initial_imu_sample.quat_wxyz
            print(f"Using IMU quaternion: {initial_imu_orientation}")
        
        # Determine the world to body rotation
        self.R_world_body = determine_world_to_body_alignment(
            initial_gt_pose, 
            initial_imu_orientation
        )
        
        # Set the initial translation (make first pose the origin)
        self.t_world_body = initial_gt_pose[:3, 3].copy()
        print(f"World to body translation: {self.t_world_body}")
        
        self.is_aligned = True
        
        # Verify alignment by transforming the first GT pose
        test_pose_body = apply_world_to_body_transform(
            initial_gt_pose, self.R_world_body, self.t_world_body
        )
        print(f"First GT pose in body frame - position: {test_pose_body[:3, 3]}")
        print(f"Should be near origin: [0, 0, 0]")
    
    def world_to_body_pose(self, pose_world):
        """Convert world frame pose to body frame"""
        if not self.is_aligned:
            raise RuntimeError("VIO system not aligned to world frame. Call align_to_world_frame first.")
        
        return apply_world_to_body_transform(pose_world, self.R_world_body, self.t_world_body)
    
    def body_to_world_pose(self, pose_body):
        """Convert body frame pose to world frame"""
        if not self.is_aligned:
            raise RuntimeError("VIO system not aligned to world frame. Call align_to_world_frame first.")
        
        return apply_body_to_world_transform(pose_body, self.R_world_body, self.t_world_body)
    def _init_backend(self, feature_type):
        if feature_type.upper() == "ORB":
            return ORBBackend()
        elif feature_type.upper() == "SIFT":
            return SIFTBackend()
        elif feature_type.upper() == "FAST":
            return FASTBackend()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def _load_calib(self, filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r
    
    def analyze_coordinate_systems(self):
        """Analyze and print coordinate system information"""
        print("\n" + "="*60)
        print("COORDINATE SYSTEM ANALYSIS")
        print("="*60)
        
        print("CAMERA CALIBRATION:")
        print(f"K_l (left camera intrinsic):\n{self.K_l}")
        print(f"P_l (left camera projection):\n{self.P_l}")
        print(f"K_r (right camera intrinsic):\n{self.K_r}")
        print(f"P_r (right camera projection):\n{self.P_r}")
        
        print(f"\nCAMERA-IMU EXTRINSICS:")
        if hasattr(self.T_cam_imu, 'rotation'):
            R_cam_imu = self.T_cam_imu.rotation().matrix()
            t_cam_imu = self.T_cam_imu.translation()
            print(f"R_cam_imu:\n{R_cam_imu}")
            print(f"t_cam_imu: {t_cam_imu}")
        else:
            print(f"T_cam_imu: {self.T_cam_imu}")
        
        print(f"\nCOORDINATE FRAMES:")
        print("Camera Frame: X=right, Y=down, Z=forward (OpenCV convention)")
        print("IMU Frame: X=forward, Y=left, Z=up")
        print("World Frame (MoCap): X=right, Z=forward, Y=up")
        print("Our 2D Plot: X=forward, Y=left")
    def _setup_gtsam(self):
        # IMU preintegration params
        I3 = np.eye(3)
        self.params = PreintegrationParams.MakeSharedU(9.81)
        sigma_a = 0.63    # m/s^2   (from 3.2 mg/√Hz)
        sigma_w = 0.035   # rad/s   (from 0.10 deg/s/√Hz
        self.params.setAccelerometerCovariance(I3 * (0.1**2))  # Increased noise
        self.params.setGyroscopeCovariance(I3 * (np.deg2rad(5)**2))  # Increased noise
        self.params.setIntegrationCovariance(I3 * (1e-4))  # Increased noise

        # Initial bias + preintegrator
        self.bias0 = imuBias.ConstantBias()
        self.accum = PreintegratedImuMeasurements(self.params, self.bias0)

        # Factor graph and solver
        self.graph = NonlinearFactorGraph()
        self.initial = Values()
        p = ISAM2Params()
        p.setRelinearizeThreshold(0.05)
        p.setFactorization("QR")
        self.isam = ISAM2(p)
        R_cl_b = np.array([
        [0, -1,  0],  # Body X (forward) -> Camera Z (forward)
        [0,  0, -1],  # Body Y (left) -> Camera -X (right)  
        [1,  0,  0]   # Body Z (up) -> Camera -Y (down)
        ], dtype=float)
        
        t_cl_b = np.array([0.06, 0.0, 0.0])  # IMU is +0.06m in camera X (right)
        
        self.T_cl_b = Pose3(Rot3(R_cl_b), Point3(*t_cl_b))  # Camera <- Body
        self.T_b_cl = self.T_cl_b.inverse()                 # Body <- Camera
        
        print("Camera-IMU extrinsics:")
        print(f"R_cl_b:\n{R_cl_b}")
        print(f"t_cl_b: {t_cl_b}")
        # State / keys / flags
        self.k = 0
        self.has_initialized_isam = False
        self.bias_key = B(0)

        # LOOSEN PRIOR NOISE MODELS
        self.pose_prior_noise = noiseModel.Diagonal.Sigmas(
            np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5], dtype=float)  # Much looser position
        )
        self.vel_prior_noise = noiseModel.Isotropic.Sigma(3, 1.0)  # Looser velocity
        self.bias_prior_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1], dtype=float)  # Much looser bias
        )
        
        # LOOSEN BETWEEN FACTOR NOISE
        self.odo_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2], dtype=float)  # Much looser
        )
        self.bias_between_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01], dtype=float)  # Looser bias between
        )

        # Camera-IMU extrinsics
        self.T_cam_imu = np.array([0.06, 0.0, 0.0])

    def initialize_isam_with_prior(self, T0_4x4=None):
        """Initialize ISAM with complete prior factors - NOW IN BODY FRAME"""
        if self.has_initialized_isam:
            return

        # Use GT initial pose if available - TRANSFORM FROM CAMERA TO BODY FRAME
        if hasattr(self, 'gt_poses') and len(self.gt_poses) > 0:
            T0_camera = self.gt_poses[0]  # GT is in camera frame
            print(f"Using GT initial CAMERA pose: position ({T0_camera[0,3]:.3f}, {T0_camera[1,3]:.3f}, {T0_camera[2,3]:.3f})")
            
            # Transform from camera frame to body frame
            T0_camera_pose3 = self._pose3_from_T(T0_camera)
            T0_body = self.T_b_cl.compose(T0_camera_pose3).compose(self.T_cl_b)
            
            body_pos = T0_body.translation()
            print(f"Transformed to BODY frame: position ({body_pos[0]:.3f}, {body_pos[1]:.3f}, {body_pos[2]:.3f})")
            
            pose0 = T0_body
        elif T0_4x4 is not None:
            # If external T0 is provided, assume it's already in body frame
            pose0 = self._pose3_from_T(T0_4x4)
            print(f"Using external BODY initial pose")
        else:
            pose0 = Pose3()
            print(f"Using identity BODY initial pose")

        # Add ALL required prior factors with stronger constraints
        self.graph.add(PriorFactorPose3(X(0), pose0, self.pose_prior_noise))
        self.graph.add(PriorFactorVector(V(0), np.zeros(3), self.vel_prior_noise))
        self.graph.add(PriorFactorConstantBias(B(0), self.bias0, self.bias_prior_noise))

        # Insert ALL initial values
        self.initial.insert(X(0), pose0)
        self.initial.insert(V(0), np.zeros(3))
        self.initial.insert(B(0), self.bias0)

        # First iSAM update
        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial.clear()
        self.k = 0
        self.has_initialized_isam = True
        print("ISAM initialized with complete prior factors IN BODY FRAME")

    def set_data(self, images_l, images_r, camera_times, gt_poses, imu_samples):
        self.images_l = images_l
        self.images_r = images_r
        self.camera_times = camera_times
        self.gt_poses = gt_poses
        self.imu_samples = imu_samples
        self.imu_times = np.array([s.timestamp for s in imu_samples]) if imu_samples else np.array([])

    def set_cam_imu_extrinsics(self, R_cl_b=None, t_cl_b=None):
        """Set camera-IMU extrinsics - OVERRIDE the default"""
        if R_cl_b is None: 
            R_cl_b = np.array([
                [0, -1,  0],
                [0,  0, -1], 
                [1,  0,  0]
            ], dtype=float)
        if t_cl_b is None: 
            t_cl_b = np.array([0.06, 0.0, 0.0])
        
        self.T_cl_b = Pose3(Rot3(R_cl_b), Point3(*t_cl_b))
        self.T_b_cl = self.T_cl_b.inverse()
        
        print("Updated Camera-IMU extrinsics:")
        print(f"R_cl_b:\n{R_cl_b}")
        print(f"t_cl_b: {t_cl_b}")

    def _pose3_from_T(self, T_4x4):
        R = Rot3(T_4x4[:3, :3])
        t = Point3(T_4x4[:3, 3])
        return Pose3(R, t)

    def _form_transf(self, R, t):
        """Makes a transformation matrix from rotation matrix and translation vector"""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_tiled_keypoints(self, img, tile_h=8, tile_w=16):
        """Splits the image into tiles and detects the 10 best keypoints in each tile"""
        def get_kps(x, y):
            impatch = img[y:y + tile_h, x:x + tile_w]
            if impatch.size == 0:
                return []
                
            # Use the backend to detect features in the patch
            pts, _ = self.backend.detect_and_describe(impatch)
            
            # Convert points back to full image coordinates
            keypoints = []
            for pt in pts:
                kp = cv2.KeyPoint()
                kp.pt = (pt[0] + x, pt[1] + y)
                keypoints.append(kp)

            return keypoints

        h, w = img.shape
        kp_list = []
        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                kps = get_kps(x, y)
                kp_list.extend(kps)
    
        return kp_list

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=1.0, max_disp=100.0):
        """Calculates the right keypoints using disparity"""
        def get_disparities(q, disp):
            disparities = np.zeros(len(q), dtype=np.float32)
            valid_mask = np.ones(len(q), dtype=bool)
            
            for i, (x, y) in enumerate(q):
                # Use rounding instead of integer casting for better precision
                x_int, y_int = int(round(x)), int(round(y))
                
                # Check bounds
                if 0 <= y_int < disp.shape[0] and 0 <= x_int < disp.shape[1]:
                    disparities[i] = disp[y_int, x_int]
                    
                    # Check if disparity is valid
                    if disparities[i] <= min_disp or disparities[i] >= max_disp or disparities[i] <= 0:
                        valid_mask[i] = False
                else:
                    valid_mask[i] = False
                    disparities[i] = 0
                    
            return disparities, valid_mask

        # Get disparities for both sets of points
        disp1_vals, mask1 = get_disparities(q1, disp1)
        disp2_vals, mask2 = get_disparities(q2, disp2)

        # Both points must be valid
        valid_mask = mask1 & mask2
        
        if np.sum(valid_mask) < 6:
            return (np.empty((0, 2), dtype=np.float32), 
                    np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32))

        # Apply the valid mask
        q1_l = q1[valid_mask]
        q2_l = q2[valid_mask]
        disp1_vals = disp1_vals[valid_mask]
        disp2_vals = disp2_vals[valid_mask]
        
        # Calculate right keypoints by subtracting disparity
        q1_r = np.copy(q1_l)
        q2_r = np.copy(q2_l)
        q1_r[:, 0] -= disp1_vals
        q2_r[:, 0] -= disp2_vals

        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """Triangulate points from both images"""
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])

        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """Calculate residuals for LM refinement"""
        r = dof[:3]
        R, _ = cv2.Rodrigues(r)
        t = dof[3:]
        transf = self._form_transf(R, t)

        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        q1_pred = Q2.dot(f_projection.T)
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]

        q2_pred = Q1.dot(b_projection.T)
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]

        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()
        return residuals

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=200):
        """Estimates the transformation matrix using RANSAC"""
        early_termination_threshold = 5
        min_error = float('inf')
        early_termination = 0

        if q1.shape[0] < 6:
            return np.eye(4)

        for _ in range(max_iter):
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            in_guess = np.zeros(6)
            try:
                opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200,
                                        args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

                error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
                error = error.reshape((Q1.shape[0] * 2, 2))
                error = np.sum(np.linalg.norm(error, axis=1))

                if error < min_error:
                    min_error = error
                    out_pose = opt_res.x
                    early_termination = 0
                else:
                    early_termination += 1
                if early_termination == early_termination_threshold:
                    break
            except:
                continue

        if min_error == float('inf'):
            return np.eye(4)

        r = out_pose[:3]
        R, _ = cv2.Rodrigues(r)
        t = out_pose[3:]
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, i):
        """Calculates the transformation matrix for the i'th frame"""
        try:
            img1_l, img2_l = self.images_l[i - 1], self.images_l[i]
            img1_r, img2_r = self.images_r[i - 1], self.images_r[i]
            
            # Get keypoints from first image
            kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)
            if len(kp1_l) == 0:
                return np.eye(4)
            
            # Get descriptors for first image
            pts1_l_full, des1_l = self.backend.detect_and_describe(img1_l)
            
            # Match features between frames
            matched_pts1, matched_pts2 = self.backend.match_features(img1_l, img2_l, pts1_l_full, des1_l)
            
            if len(matched_pts1) < 6:
                return np.eye(4)
            disp1 = self.disparity.compute(img1_l, img1_r).astype(np.float32) / 16.0
            disp2 = self.disparity.compute(img2_l, img2_r).astype(np.float32) / 16.0
            # Compute disparities - FIXED: Check if we need to compute previous disparity
            if i-1 >= len(self.disparities):
                self.disparities.append(disp1)
            self.disparities.append(disp2)

            # Calculate right keypoints using disparity
            q1_l, q1_r, q2_l, q2_r = self.calculate_right_qs(matched_pts1, matched_pts2, disp1, disp2)
            
            
            if len(q1_l) < 6:
                return np.eye(4)

            # Triangulate 3D points
            Q1, Q2 = self.calc_3d(q1_l, q1_r, q2_l, q2_r)
            
            # Estimate pose
            transformation_matrix = self.estimate_pose(q1_l, q2_l, Q1, Q2)
            # Check if we got a valid transformation
            translation_norm = np.linalg.norm(transformation_matrix[:3, 3])
            R = transformation_matrix[:3, :3]
            t = transformation_matrix[:3, 3]
            rotation_angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
            if translation_norm < 0.001 and rotation_angle < np.deg2rad(0.3):
                return None
                
            if translation_norm > 1.0 or rotation_angle > np.deg2rad(45):
                return None
            return transformation_matrix
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return np.eye(4)

    def preintegrate_between(self, preint, t0, t1, imu_samples, imu_times):
        """Preintegrate IMU between two timestamps"""
        assert t1 > t0, "t1 must be > t0"
        preint.resetIntegration()

        # Find IMU samples in interval
        lo = np.searchsorted(imu_times, t0, side="right")
        hi = np.searchsorted(imu_times, t1, side="right")

        if lo >= len(imu_times) or hi > len(imu_times) or lo >= hi:
            return

        # Integrate measurements
        integrated_any = False
        for k in range(lo, hi):
            s = imu_samples[k]
            if k == lo:
                dt = s.timestamp - t0
            else:
                dt = s.timestamp - imu_samples[k-1].timestamp
            
            if dt > 0 and dt < 0.1:  # Reasonable dt check
                preint.integrateMeasurement(s.accel, s.gyro, dt)
                integrated_any = True
        
        if not integrated_any:
            print(f"  No valid IMU integration between {t0:.3f} and {t1:.3f}")

    def fuse_vo_imu_step(self, T_km1_k_4x4, t_km1=None, t_k=None):
        """Fuse VO and IMU - COMPLETE VERSION"""
        if not self.has_initialized_isam:
            self.initialize_isam_with_prior()

        k = self.k + 1

        # Convert VO delta from CAMERA frame to Pose3
        try:
            R_cl = Rot3(T_km1_k_4x4[:3, :3])
            t_cl = Point3(*T_km1_k_4x4[:3, 3])
            T_meas_cl = Pose3(R_cl, t_cl)
        except:
            T_meas_cl = Pose3()

        
        # PROPER transformation from camera to body frame
        T_meas_body = self.T_cl_b.inverse().compose(T_meas_cl).compose(self.T_cl_b)
        

        # Add IMU factor if we have samples and valid timestamps
        added_imu = False
        if (t_km1 is not None) and (t_k is not None) and (len(self.imu_samples) > 0) and (t_k > t_km1):
            try:
                self.preintegrate_between(self.accum, t_km1, t_k, self.imu_samples, self.imu_times)
                if self.accum.deltaTij() > 0.001:
                    self.graph.add(ImuFactor(X(k-1), V(k-1), X(k), V(k), self.bias_key, self.accum))
                    added_imu = True
            except Exception as e:
                print(f"IMU preintegration failed at frame {k}: {e}")

        # BIAS CHAINING - Important for maintaining orientation constraints
        if k % 10 == 0:  # Every 10 frames
            new_bias_key = B(k // 10)
            self.graph.add(BetweenFactorConstantBias(
                self.bias_key, new_bias_key,
                imuBias.ConstantBias(),  # Zero bias change
                self.bias_between_noise  # Tight constraints
            ))
            self.initial.insert(new_bias_key, self.bias0)  # Initialize with same bias
            self.bias_key = new_bias_key

        # Add VO factor - IN BODY FRAME
        self.graph.add(BetweenFactorPose3(X(k-1), X(k), T_meas_body, self.odo_noise))

        # Initialize new body pose
        if not self.initial.exists(X(k)):
            try:
                prev_body_pose = self.isam.calculateEstimate().atPose3(X(k - 1))
                self.initial.insert(X(k), prev_body_pose.compose(T_meas_body))
            except:
                self.initial.insert(X(k), Pose3())
        
        if not self.initial.exists(V(k)):
            self.initial.insert(V(k), np.array([0.1, 0.0, 0.0]))  # Small initial velocity

        try:
            self.isam.update(self.graph, self.initial)
            result = self.isam.calculateEstimate()
            
            current_pose = result.atPose3(X(k))
            if k % 10 == 0:
                body_pose = result.atPose3(X(k))
                pos = body_pose.translation()
                R = body_pose.rotation().matrix()
                yaw = np.arctan2(R[1,0], R[0,0])
                
                prev_body_pose = result.atPose3(X(k-1))
                pos_change = body_pose.translation() - prev_body_pose.translation()
            
            self.graph = NonlinearFactorGraph()
            self.initial.clear()
            self.k = k
        
            return current_pose
            
        except Exception as e:
            # Reset and return identity
            self.graph = NonlinearFactorGraph()
            self.initial.clear()
            return Pose3()


    def run(self, max_frames=None, debug=False):
        """Run the complete VO pipeline with IMU fusion"""
        if max_frames is None:
            max_frames = len(self.images_l)
        max_frames = min(max_frames, len(self.images_l))
        
        print(f"Running VO+IMU on {max_frames} frames...")
        
        # Initialize with first frame
        self.initialize_isam_with_prior()
        
        # Process frames
        for i in tqdm(range(1, max_frames), desc="Processing"):
            try:
                # Get VO pose estimate
                T_vo = self.get_pose(i)
                
                # Get timestamps - ensure they're valid
                t_km1 = self.camera_times[i-1] if i < len(self.camera_times) else None
                t_k = self.camera_times[i] if i < len(self.camera_times) else None
                
                # Validate timestamps for IMU integration
                if t_km1 is not None and t_k is not None and t_k <= t_km1:
                    print(f"Frame {i}: Invalid timestamps {t_km1:.3f} -> {t_k:.3f}")
                    t_km1 = t_k = None
                
                # Fuse with IMU
                est_pose = self.fuse_vo_imu_step(T_vo, t_km1, t_k)
                self.estimated_poses.append(est_pose)
                
                if debug and i % 5 == 0:
                    try:
                        result = self.isam.calculateEstimate()
                        current_pose = result.atPose3(X(self.k))
                        
                        # Extract all components
                        translation = current_pose.translation()
                        rotation = current_pose.rotation()
                        
                        # Get rotation as matrix and Euler angles
                        R = rotation.matrix()
                        
                        # Calculate Euler angles (properly)
                        # GTSAM uses different conventions, let's be explicit
                        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
                        singular = sy < 1e-6
                        
                        if not singular:
                            roll = np.arctan2(R[2,1], R[2,2])
                            pitch = np.arctan2(-R[2,0], sy)
                            yaw = np.arctan2(R[1,0], R[0,0])
                        else:
                            roll = np.arctan2(-R[1,2], R[1,1])
                            pitch = np.arctan2(-R[2,0], sy)
                            yaw = 0
                        
                        print(f"\nFrame {i} - FULL POSE ANALYSIS:")
                        print(f"  Translation: X={translation[0]:.3f}, Y={translation[1]:.3f}, Z={translation[2]:.3f}")
                        print(f"  Rotation Euler: Roll={np.degrees(roll):.1f}°, Pitch={np.degrees(pitch):.1f}°, Yaw={np.degrees(yaw):.1f}°")
                        print(f"  Rotation Matrix:")
                        print(f"    [{R[0,0]:.3f}, {R[0,1]:.3f}, {R[0,2]:.3f}]")
                        print(f"    [{R[1,0]:.3f}, {R[1,1]:.3f}, {R[1,2]:.3f}]")
                        print(f"    [{R[2,0]:.3f}, {R[2,1]:.3f}, {R[2,2]:.3f}]")
                        
                        # Also print the raw VO input for comparison
                        if T_vo is not None:
                            vo_trans = T_vo[:3, 3]
                            vo_roll = np.arctan2(T_vo[2,1], T_vo[2,2])
                            vo_pitch = np.arctan2(-T_vo[2,0], np.sqrt(T_vo[2,1]**2 + T_vo[2,2]**2))
                            vo_yaw = np.arctan2(T_vo[1,0], T_vo[0,0])
                            print(f"  VO Input: Trans({vo_trans[0]:.3f}, {vo_trans[1]:.3f}, {vo_trans[2]:.3f}), Rot({np.degrees(vo_roll):.1f}, {np.degrees(vo_pitch):.1f}, {np.degrees(vo_yaw):.1f})°")
                        
                    except Exception as e:
                        print(f"  Could not extract pose details: {e}")
                    
            except Exception as e:
                print(f"Error at frame {i}: {e}")
                # Safer fallback
                try:
                    fallback_pose = Pose3()  # Identity pose as fallback
                    self.estimated_poses.append(fallback_pose)
                except:
                    # Ultimate fallback
                    self.estimated_poses.append(Pose3())
        
        print(f"Completed {len(self.estimated_poses)} frames")
        return self.estimated_poses

    def evaluate(self):
        """Evaluate against ground truth with proper coordinate comparison"""
        if len(self.gt_poses) == 0:
            print("No ground truth for evaluation")
            return None
            
        errors = []
        est_positions = []
        gt_positions = []
        
        min_len = min(len(self.estimated_poses), len(self.gt_poses))
        
        for i in range(min_len):
            est_pos = self.estimated_poses[i].translation()
            gt_pos = self.gt_poses[i][:3, 3]
            
            # Compare in same coordinate frame
            error = np.linalg.norm(est_pos - gt_pos)
            errors.append(error)
            
            # Store for plotting: Camera frame (X-forward, Y-left, Z-up)
            est_positions.append((est_pos[0], est_pos[1]))  # X-forward vs Y-left
            gt_positions.append((gt_pos[0], gt_pos[1]))     # X-forward vs Y-left
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"Trajectory Evaluation:")
        print(f"  Average error: {avg_error:.3f} m")
        print(f"  Maximum error: {max_error:.3f} m")
        print(f"  Frames evaluated: {min_len}")
        
        return {
            'errors': errors,
            'est_positions': est_positions,
            'gt_positions': gt_positions,
            'avg_error': avg_error,
            'max_error': max_error
        }

def main():
    print("=== STEREO VISUAL ODOMETRY WITH IMU FUSION (PROPER WORLD ALIGNMENT) ===\n")
    
    # Initialize data loader
    data_loader = DataLoader()
    
    # Load all data
    print("1. LOADING DATA...")
    mocap_times, gt_poses_world, mocap_df = data_loader.load_mocap_data(MOCAP_CSV)
    imu_samples = data_loader.load_imu_data(IMU_CSV)
    camera_times, images_l, images_r = data_loader.load_camera_data(LEFT_DATACSV, LEFT_DIR, RIGHT_DIR)
    
    print(f"Loaded: {len(images_l)} images, {len(gt_poses_world)} GT poses, {len(imu_samples)} IMU samples")

    # Use the actual number of camera frames we have
    num_frames = len(images_l)
    print(f"Using {num_frames} camera frames")
    
    # Subsample GT poses to match camera frame rate if needed
    if len(gt_poses_world) > num_frames:
        step = len(gt_poses_world) // num_frames
        gt_poses_world = gt_poses_world[::step][:num_frames]
        print(f"Subsampled GT poses to {len(gt_poses_world)}")
    
    print("\n2. WORLD FRAME ALIGNMENT...")
    
    # Initialize VO system
    vo = StereoVO(CALIB_PATH, FEATURE_TYPE)
    vo.set_cam_imu_extrinsics()
    
    # Get initial IMU sample for alignment (first sample near camera start)
    initial_imu_sample = None
    if imu_samples and camera_times:
        # Find IMU sample closest to first camera timestamp
        first_camera_time = camera_times[0]
        time_diffs = [abs(s.timestamp - first_camera_time) for s in imu_samples]
        closest_imu_idx = np.argmin(time_diffs)
        if time_diffs[closest_imu_idx] < 0.1:  # Within 100ms
            initial_imu_sample = imu_samples[closest_imu_idx]
            print(f"Using IMU sample at t={initial_imu_sample.timestamp:.3f}s for alignment")
    
    # Perform world frame alignment using first GT pose
    gt_poses_body = []
    if gt_poses_world:
        # Print initial pose info for debugging
        initial_gt_pose = gt_poses_world[0]
        initial_pos = initial_gt_pose[:3, 3]
        initial_rot = initial_gt_pose[:3, :3]
        rx, ry, rz = rotation_matrix_to_euler(initial_rot)
        
        print(f"Initial GT pose in world frame:")
        print(f"  Position: X(left/right)={initial_pos[0]:.3f}, Y(up/down)={initial_pos[1]:.3f}, Z(forward/back)={initial_pos[2]:.3f}")
        print(f"  Rotation: Roll={np.degrees(rx):.1f}°, Pitch={np.degrees(ry):.1f}°, Yaw={np.degrees(rz):.1f}°")
        
        vo.align_to_world_frame(initial_gt_pose, initial_imu_sample)
        
        # Set the translation to make first pose the origin in body frame
        vo.t_world_body = initial_gt_pose[:3, 3].copy()
        print(f"World to body translation set to: {vo.t_world_body}")
        
        # Transform all GT poses to body frame
        gt_poses_body = [vo.world_to_body_pose(T) for T in gt_poses_world]
        print(f"Transformed {len(gt_poses_body)} GT poses to body frame")
        
        # Verify alignment - first pose should be near origin
        first_pose_body = gt_poses_body[0]
        body_pos = first_pose_body[:3, 3]
        body_rot = first_pose_body[:3, :3]
        rx_body, ry_body, rz_body = rotation_matrix_to_euler(body_rot)
        
        print(f"First GT pose in body frame:")
        print(f"  Position: X(forward)={body_pos[0]:.3f}, Y(left/right)={body_pos[1]:.3f}, Z(up/down)={body_pos[2]:.3f}")
        print(f"  Rotation: Roll={np.degrees(rx_body):.1f}°, Pitch={np.degrees(ry_body):.1f}°, Yaw={np.degrees(rz_body):.1f}°")
        print(f"  Should be near origin: [0, 0, 0]")
        
        # Test coordinate transformation makes sense
        print(f"\nCoordinate transformation check:")
        print(f"  MoCap Z(forward)={initial_pos[2]:.3f} -> Body X(forward)={body_pos[0]:.3f}")
        print(f"  MoCap X(left/right)={initial_pos[0]:.3f} -> Body Y(left/right)={body_pos[1]:.3f}")
        print(f"  MoCap Y(up/down)={initial_pos[1]:.3f} -> Body Z(up/down)={body_pos[2]:.3f}")
        
        # Calculate GT path length in world frame for reference
        gt_path_length = 0
        for i in range(1, len(gt_poses_world)):
            pos1 = gt_poses_world[i-1][:3, 3]
            pos2 = gt_poses_world[i][:3, 3]
            gt_path_length += np.linalg.norm(pos2 - pos1)
        print(f"GT path length in world frame: {gt_path_length:.2f}m")
        
        # Additional verification: test a few more poses
        print(f"\nTransformation consistency check (first 3 poses):")
        for i in range(min(3, len(gt_poses_world))):
            world_pose = gt_poses_world[i]
            body_pose = gt_poses_body[i]
            
            world_pos = world_pose[:3, 3]
            body_pos = body_pose[:3, 3]
            
            print(f"  Pose {i}: MoCap(Z,X)=({world_pos[2]:.2f}, {world_pos[0]:.2f}) -> Body(X,Y)=({body_pos[0]:.2f}, {body_pos[1]:.2f})")
    else:
        print("No GT poses available for alignment")
    # Ensure consistent lengths
    min_len = min(num_frames, len(gt_poses_body) if gt_poses_body else num_frames)
    
    images_l = images_l[:min_len]
    images_r = images_r[:min_len]
    camera_times = camera_times[:min_len] if len(camera_times) >= min_len else [i * 0.1 for i in range(min_len)]
    gt_poses_body = gt_poses_body[:min_len] if gt_poses_body else []
    
    print(f"Final aligned: {min_len} frames")
    
    # Set data with body-frame GT poses
    vo.set_data(images_l, images_r, camera_times, gt_poses_body, imu_samples)
    
    # Run VIO (estimates will be in body frame) in_len
    print(f"\n3. RUNNING VISUAL ODOMETRY WITH IMU FUSION...")
    estimated_poses_body = vo.run(max_frames=min_len, debug=False)
    
    print(f"Completed VO processing for {len(estimated_poses_body)} frames")
    
    # Convert estimated poses back to world frame for evaluation
    print(f"\n4. CONVERTING TO WORLD FRAME FOR EVALUATION...")
    estimated_poses_world = []
    if vo.is_aligned and gt_poses_body:
        for i, pose_body in enumerate(estimated_poses_body):
            if hasattr(pose_body, 'matrix'):
                # GTSAM Pose3 to numpy
                pose_body_np = pose_body.matrix()
            else:
                pose_body_np = pose_body
                
            pose_world = vo.body_to_world_pose(pose_body_np)
            estimated_poses_world.append(pose_world)
            
            # Print first few poses for verification
            if i < 3:
                pos_world = pose_world[:3, 3]
                rot_world = pose_world[:3, :3]
                rx, ry, rz = rotation_matrix_to_euler(rot_world)
                print(f"  Frame {i}: World position=({pos_world[0]:.3f}, {pos_world[1]:.3f}, {pos_world[2]:.3f}), "
                      f"Yaw={np.degrees(rz):.1f}°")
        
        print(f"Converted {len(estimated_poses_world)} poses to world frame")
    else:
        print("Cannot convert to world frame - system not properly aligned")
        estimated_poses_world = []
    
    # Evaluate trajectory accuracy
    print(f"\n5. TRAJECTORY EVALUATION...")
    if estimated_poses_world and gt_poses_world:
        # Ensure same length for evaluation
        eval_len = min(len(estimated_poses_world), len(gt_poses_world))
        estimated_poses_world_eval = estimated_poses_world[:eval_len]
        gt_poses_eval = gt_poses_world[:eval_len]
        
        print(f"Evaluating {eval_len} frames...")
        print("\n--- COMBINED UMEYAMA + RELATIVE ALIGNMENT ---")
        est_relative, gt_relative, errors, umeyama_aligned_poses = plot_umeyama_relative_trajectory(
            estimated_poses_world_eval, gt_poses_eval
        )
        # METHOD 1: Relative motion evaluation (recommended)
        print("\n--- RELATIVE MOTION EVALUATION ---")
        relative_metrics = evaluate_using_relative_motion(estimated_poses_world_eval, gt_poses_eval)
        
        # METHOD 2: Traditional absolute evaluation (for comparison)
        print("\n--- ABSOLUTE POSE EVALUATION ---")
        absolute_errors = []
        absolute_orientation_errors = []
        
        for i in range(eval_len):
            # Position error
            est_pos = estimated_poses_world_eval[i][:3, 3]
            gt_pos = gt_poses_eval[i][:3, 3]
            pos_error = np.linalg.norm(est_pos - gt_pos)
            absolute_errors.append(pos_error)
            
            # Orientation error (using rotation matrix, not Euler angles)
            est_rot = estimated_poses_world_eval[i][:3, :3]
            gt_rot = gt_poses_eval[i][:3, :3]
            R_diff = est_rot.T @ gt_rot
            orient_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
            absolute_orientation_errors.append(orient_error)
        
        avg_abs_error = np.mean(absolute_errors)
        max_abs_error = np.max(absolute_errors)
        avg_abs_orient_error = np.degrees(np.mean(absolute_orientation_errors))
        
        print(f"Absolute Pose Evaluation ({eval_len} frames):")
        print(f"  Average position error: {avg_abs_error:.3f} m")
        print(f"  Maximum position error: {max_abs_error:.3f} m")
        print(f"  Average orientation error: {avg_abs_orient_error:.1f}°")
        
        # Store both metrics for visualization
        error_metrics = {
            'relative': relative_metrics,
            'absolute': {
                'position_errors': absolute_errors,
                'orientation_errors': absolute_orientation_errors,
                'avg_position_error': avg_abs_error,
                'avg_orientation_error': avg_abs_orient_error
            }
        }
        save_umeyama_relative_results(est_relative, gt_relative, errors, "umeyama_relative_results.csv")
    else:
        print("  Cannot evaluate - missing GT or estimated poses")
        error_metrics = None
    
    
    # VISUALIZATION - 2D PLOTS
    print("\n8. VISUALIZING 2D TRAJECTORIES...")
    
    # Prepare data for plotting
    est_traj_2d = []
    gt_traj_2d = []
    
    # Extract 2D trajectories (X-Z plane for world frame)
    for pose in estimated_poses_body:
        if hasattr(pose, 'matrix'):
            pos = pose.translation()
        else:
            pos = pose_body[:3, 3]
        est_traj_2d.append([pos[2], pos[0]])  # X and Z coordinates
    for pose in gt_poses_body[:len(estimated_poses_body)]:
        pos = pose[:3, 3]
        gt_traj_2d.append([pos[2], pos[0]])  # X and Z coordinates
    
    est_traj_2d = np.array(est_traj_2d)
    gt_traj_2d = np.array(gt_traj_2d)
    if estimated_poses_world and gt_poses_world:
        est_relative, gt_relative = plot_relative_trajectory(estimated_poses_world, gt_poses_world[:len(estimated_poses_world)])
        # Calculate error on aligned trajectories
        errors = np.linalg.norm(est_relative - gt_relative, axis=1)
        avg_error = np.mean(errors)
        print(f"Average error after relative alignment: {avg_error:.3f}m")
        print(f"\n6. SAVING RESULTS...")
        save_relative_trajectory_results(est_relative, gt_relative, errors, "relative_trajectory_results.csv")
    # ALTERNATIVE: Use Umeyama alignment
    if estimated_poses_world and gt_poses_world:
        aligned_poses = align_trajectory_umeyama(estimated_poses_world, gt_poses_world[:len(estimated_poses_world)])
        
        # Plot aligned trajectories
        est_aligned = np.array([pose[:3, 3] for pose in aligned_poses])
        gt_positions = np.array([pose[:3, 3] for pose in gt_poses_world[:len(aligned_poses)]])
        save_umeyama_trajectory_results(est_aligned, gt_positions, errors, "Umeyama_relative_trajectory_results.csv")
        plt.figure(figsize=(10, 8))
        plt.plot(est_aligned[:, 0], est_aligned[:, 2], 'b-', label='Estimated (Aligned)', linewidth=2)
        plt.plot(gt_positions[:, 0], gt_positions[:, 2], 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        plt.title('Trajectory After Umeyama Alignment')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        
        # Calculate final errors
        final_errors = np.linalg.norm(est_aligned - gt_positions, axis=1)
        final_avg_error = np.mean(final_errors)
        print(f"Average error after Umeyama alignment: {final_avg_error:.3f}m")
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: 2D Trajectory Comparison
    plt.subplot(2, 2, 1)
    plt.plot(est_traj_2d[:, 0], est_traj_2d[:, 1], 'b-', label='Estimated', linewidth=2)
    plt.plot(est_traj_2d[0, 0], est_traj_2d[0, 1], 'go', markersize=8, label='Start')
    plt.plot(est_traj_2d[-1, 0], est_traj_2d[-1, 1], 'bo', markersize=8, label='End')
    
    if len(gt_traj_2d) > 0:
        plt.plot(gt_traj_2d[:, 0], gt_traj_2d[:, 1], 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
        plt.plot(gt_traj_2d[0, 0], gt_traj_2d[0, 1], 'go', markersize=8)
        plt.plot(gt_traj_2d[-1, 0], gt_traj_2d[-1, 1], 'ro', markersize=8)
    
    plt.xlabel('X [m] (World Frame)')
    plt.ylabel('Z [m] (World Frame)')
    plt.title(f'2D Trajectory - World Frame\n{FEATURE_TYPE} + IMU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(2, 2, 2)
    if error_metrics and error_metrics['relative']:
        rel_errors = error_metrics['relative']['position_errors']
        plt.plot(rel_errors, 'g-', linewidth=1, alpha=0.7)
        plt.axhline(y=error_metrics['relative']['avg_position_error'], color='r', linestyle='--', 
                    label=f'Average: {error_metrics["relative"]["avg_position_error"]:.3f}m/step')
        plt.xlabel('Step')
        plt.ylabel('Relative Position Error (m)')
        plt.title('Relative Position Error per Step')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 3: Relative orientation errors  
    plt.subplot(2, 2, 3)
    if error_metrics and error_metrics['relative']:
        rel_orient_errors = np.degrees(error_metrics['relative']['orientation_errors'])
        plt.plot(rel_orient_errors, 'm-', linewidth=1, alpha=0.7)
        plt.axhline(y=error_metrics['relative']['avg_orientation_error'], color='r', linestyle='--',
                    label=f'Average: {error_metrics["relative"]["avg_orientation_error"]:.1f}°/step')
        plt.xlabel('Step')
        plt.ylabel('Relative Orientation Error (°)')
        plt.title('Relative Orientation Error per Step')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    plt.subplot(2, 2, 4)
    stats_text = "Performance Summary:\n"
    if error_metrics and error_metrics['relative']:
        stats_text += f"Relative Motion:\n"
        stats_text += f"  Pos: {error_metrics['relative']['avg_position_error']:.3f}m/step\n"
        stats_text += f"  Orient: {error_metrics['relative']['avg_orientation_error']:.1f}°/step\n"
    if error_metrics and error_metrics['absolute']:
        stats_text += f"Absolute Pose:\n"
        stats_text += f"  Pos: {error_metrics['absolute']['avg_position_error']:.3f}m\n"
        stats_text += f"  Orient: {error_metrics['absolute']['avg_orientation_error']:.1f}°\n"
    if 'gt_path_length' in locals():
        stats_text += f"Path Length: {gt_path_length:.1f}m"

    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.axis('off')
    plt.title('Performance Summary')
    
    plt.tight_layout()
    plt.show()

# Make sure to add this import at the top of your vio.py file if not already present
# from plot_mocap_xy import MocapHelper
# SAVING TRAJECTORIES
def save_relative_trajectory_results(est_relative, gt_relative, errors, csv_filename="relative_trajectory_results.csv"):
    """
    Save relative trajectory positions for plot_relative_trajectory (2D data)
    """
    try:
        combined_data = []
        
        for i in range(len(est_relative)):
            row = {
                'frame': i,
                'est_x': -est_relative[i, 1],  # Same as plot: -y
                'est_y': -est_relative[i, 0],  # Same as plot: -x
                'gt_x': gt_relative[i, 0],     # Same as plot: x
                'gt_y': gt_relative[i, 2],     # Same as plot: z
                'position_error_m': errors[i]
            }
            combined_data.append(row)
        
        df = pd.DataFrame(combined_data)
        df.to_csv(csv_filename, index=False)
        print(f"Relative trajectory data saved to: {csv_filename}")
        return df
        
    except Exception as e:
        print(f"Error saving relative trajectories: {e}")
        return None

def save_umeyama_trajectory_results(est_aligned, gt_positions, errors, csv_filename="umeyama_trajectory_results.csv"):
    """
    Save Umeyama aligned trajectories (3D poses)
    """
    try:
        combined_data = []
        
        for i in range(len(est_aligned)):
            row = {
                'frame': i,
                'est_x': est_aligned[i, 0],  # X coordinate
                'est_y': est_aligned[i, 2],  # Z coordinate (for top-down view)
                'gt_x': gt_positions[i, 0],  # X coordinate
                'gt_y': gt_positions[i, 2],  # Z coordinate (for top-down view)
                'position_error_m': errors[i]
            }
            combined_data.append(row)
        
        df = pd.DataFrame(combined_data)
        df.to_csv(csv_filename, index=False)
        print(f"Umeyama trajectory data saved to: {csv_filename}")
        return df
        
    except Exception as e:
        print(f"Error saving Umeyama trajectories: {e}")
        return None
def align_trajectory_umeyama_then_relative(estimated_poses, gt_poses):
    """
    First apply Umeyama alignment for optimal global transformation,
    then make trajectories relative to their starting points
    """
    # Step 1: Apply Umeyama alignment
    T_align = umeyama_alignment(estimated_poses, gt_poses)
    
    # Apply Umeyama transformation to all estimated poses
    umeyama_aligned_poses = []
    for pose in estimated_poses:
        aligned_pose = T_align @ pose
        umeyama_aligned_poses.append(aligned_pose)
    
    # Step 2: Make both trajectories relative to their starting points
    est_positions = np.array([pose[:3, 3] for pose in umeyama_aligned_poses])
    gt_positions = np.array([pose[:3, 3] for pose in gt_poses])
    
    # Make relative to first pose
    est_relative = est_positions - est_positions[0]
    gt_relative = gt_positions - gt_positions[0]
    
    return est_relative, gt_relative, umeyama_aligned_poses

def plot_umeyama_relative_trajectory(estimated_poses, gt_poses):
    """Plot trajectories after Umeyama alignment and relative transformation"""
    
    est_relative, gt_relative, umeyama_aligned_poses = align_trajectory_umeyama_then_relative(
        estimated_poses, gt_poses
    )
    
    # Create comprehensive visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Relative trajectories (top-down view)
    plt.subplot(1, 3, 1)
    plt.plot(est_relative[:, 0], est_relative[:, 2], 'b-', label='Estimated (Umeyama+Relative)', linewidth=2)
    plt.plot(gt_relative[:, 0], gt_relative[:, 2], 'r-', label='Ground Truth (Relative)', linewidth=2, alpha=0.7)
    plt.plot(est_relative[0, 0], est_relative[0, 2], 'go', markersize=8, label='Start')
    plt.plot(est_relative[-1, 0], est_relative[-1, 2], 'bo', markersize=8, label='End Est')
    plt.plot(gt_relative[-1, 0], gt_relative[-1, 2], 'ro', markersize=8, label='End GT')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Umeyama + Relative Alignment\n(Top-Down View)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 2: Absolute trajectories after Umeyama only
    plt.subplot(1, 3, 2)
    est_abs = np.array([pose[:3, 3] for pose in umeyama_aligned_poses])
    gt_abs = np.array([pose[:3, 3] for pose in gt_poses])
    
    plt.plot(est_abs[:, 0], est_abs[:, 2], 'b-', label='Estimated (Umeyama)', linewidth=2)
    plt.plot(gt_abs[:, 0], gt_abs[:, 2], 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Umeyama Alignment Only\n(Absolute Coordinates)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Plot 3: Error progression
    plt.subplot(1, 3, 3)
    errors = np.linalg.norm(est_relative - gt_relative, axis=1)
    cumulative_distance = np.cumsum(np.linalg.norm(np.diff(gt_relative, axis=0), axis=1))
    cumulative_distance = np.insert(cumulative_distance, 0, 0)  # Start at 0
    
    plt.plot(cumulative_distance, errors, 'g-', linewidth=2)
    plt.xlabel('Distance Traveled [m]')
    plt.ylabel('Position Error [m]')
    plt.title('Error vs Distance Traveled')
    plt.grid(True)
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    final_error = errors[-1]
    
    plt.text(0.05, 0.95, f'Avg Error: {avg_error:.3f}m\nMax Error: {max_error:.3f}m\nFinal Error: {final_error:.3f}m', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive metrics
    print(f"\n=== UMEYAMA + RELATIVE ALIGNMENT METRICS ===")
    print(f"Average position error: {avg_error:.3f} m")
    print(f"Maximum position error: {max_error:.3f} m") 
    print(f"Final position error: {final_error:.3f} m")
    print(f"Total distance traveled: {cumulative_distance[-1]:.2f} m")
    print(f"Average error as % of path: {(avg_error/cumulative_distance[-1])*100:.1f}%")
    
    return est_relative, gt_relative, errors, umeyama_aligned_poses
def save_umeyama_relative_results(est_relative, gt_relative, errors, csv_filename="umeyama_relative_results.csv"):
    """Save Umeyama + Relative alignment results"""
    try:
        combined_data = []
        
        for i in range(len(est_relative)):
            row = {
                'frame': i,
                'est_x': est_relative[i, 0],  # X coordinate after alignment
                'est_y': est_relative[i, 2],  # Z coordinate (for top-down view)
                'gt_x': gt_relative[i, 0],    # X coordinate
                'gt_y': gt_relative[i, 2],    # Z coordinate
                'position_error_m': errors[i],
                'cumulative_distance': np.sum(np.linalg.norm(np.diff(gt_relative[:i+1], axis=0), axis=1)) if i > 0 else 0
            }
            combined_data.append(row)
        
        df = pd.DataFrame(combined_data)
        df.to_csv(csv_filename, index=False)
        print(f"Umeyama + Relative results saved to: {csv_filename}")
        return df
        
    except Exception as e:
        print(f"Error saving Umeyama + Relative results: {e}")
        return None
if __name__ == "__main__":
    main()