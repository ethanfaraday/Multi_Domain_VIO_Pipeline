import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from gtsam import (
    ISAM2, NonlinearFactorGraph, Values, Pose3, Point3, Rot3,
    PreintegrationParams, PreintegratedImuMeasurements,
    PriorFactorPose3, PriorFactorVector, PriorFactorConstantBias,
    BetweenFactorConstantBias, ImuFactor, noiseModel, imuBias, BetweenFactorPose3)
from visualization import plotting
from visualization.video import play_trip
from bag_of_words import BoW, make_stackimage
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from gtsam.symbol_shorthand import X, V, B
from dataclasses import dataclass
from typing import List, Tuple
from gtsam.utils import plot as gplot
from gtsam import Marginals
import gtsam
import pandas as pd
import csv
import glob, math, json
from plot_mocap_xy import MocapHelper

############################################# FEATURE TYPE ############################################
FEATURE_TYPE = "ORB"  # ORB, SIFT, or defaulted as FAST
#######################################################################################################

# -------------------------
# YOUR ABSOLUTE DATA PATHS
# -------------------------
CALIB_PATH   = r"C:\VIOCODE\ComputerVision\VisualOdometry\calib.txt"

MOCAP_CSV    = Path(r"C:\VIOCODE\ComputerVision\VisualOdometry\MOCAP\Husky_squiggle.csv")
LEFT_DIR     = r"C:\VIOCODE\ComputerVision\VisualOdometry\squigle\left_image"
RIGHT_DIR    = r"C:\VIOCODE\ComputerVision\VisualOdometry\squigle\right_image"
LEFT_DATACSV = r"C:\VIOCODE\ComputerVision\VisualOdometry\squigle\left_image\data.csv"
RIGHT_DATACSV= r"C:\VIOCODE\ComputerVision\VisualOdometry\squigle\right_image\data.csv"  # (not used, but here for completeness)
IMU_CSV      = r"C:\VIOCODE\ComputerVision\VisualOdometry\squigle\imu\imu_data.csv"

# If you have a poses.txt somewhere, point it here; otherwise leave None
POSES_PATH   = None  # e.g., r"C:\VIOCODE\ComputerVision\VisualOdometry\straight_line\poses.txt"

# swap x <-> z
A_swap = np.array([[0,0,1],
                   [0,1,0],
                   [1,0,0]], dtype=float)
A_zed_to_mocap = np.array([
    [0, 0, 1],   # X_mocap = Z_zed
    [0, 1, 0],   # Y stays (optional)
    [-1, 0, 0]   # Z_mocap = -X_zed
])
# optional flips if your GT has opposite sign on an axis
Fx = np.diag([-1,1,1])    # flip x
Fz = np.diag([ 1,1,-1])   # flip z

A_plot = A_swap
def remap_pose(T, A=A_zed_to_mocap):
    """Map a 4x4 pose T with axis transform A (3x3)."""
    R = T[:3, :3]; t = T[:3, 3]
    Rm = A @ R @ A.T
    tm = A @ t
    Tp = np.eye(4)
    Tp[:3, :3] = Rm
    Tp[:3, 3]  = tm
    return Tp  
# DEBUGGING IMPLEMENTATION
def debug_feature_matching(vo, backend, frame_idx=0):
    """Debug feature matching between consecutive frames"""
    img1 = vo.images_l[frame_idx]
    img2 = vo.images_l[frame_idx + 1]
    
    # Detect and describe features
    pts1, des1 = backend.detect_and_describe(img1)
    pts2, des2 = backend.detect_and_describe(img2)
    
    print(f"Frame {frame_idx}: Found {len(pts1)} keypoints in img1, {len(pts2)} in img2")
    
    # Match features
    idx1, idx2, good_matches = backend.match(des1, des2)
    
    print(f"Found {len(good_matches)} good matches")
    
    # Visualize matches
    if len(good_matches) > 0:
        # Create a visualization
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Draw matches
        match_img = cv2.drawMatches(
            img1_color, [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in pts1],
            img2_color, [cv2.KeyPoint(x=p[0], y=p[1], size=10) for p in pts2],
            good_matches[:50], None, flags=2
        )
        
        cv2.imshow("Feature Matches", match_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return len(good_matches), pts1, pts2, idx1, idx2
def debug_imu_preintegration(vo, frame_idx=1):
    """Debug IMU preintegration between frames"""
    if len(vo.imu) == 0:
        print("No IMU data available")
        return
    
    t_km1 = vo.cam_times[frame_idx - 1]
    t_k = vo.cam_times[frame_idx]
    
    print(f"Camera times: t_{frame_idx-1}={t_km1:.3f}s, t_{frame_idx}={t_k:.3f}s")
    print(f"Time delta: {t_k - t_km1:.3f}s")
    
    # Find IMU samples in this interval
    imu_in_interval = [s for s in vo.imu if t_km1 < s.timestamp <= t_k]
    print(f"Found {len(imu_in_interval)} IMU samples in interval")
    
    if len(imu_in_interval) > 0:
        # Print first few samples
        for i, sample in enumerate(imu_in_interval[:3]):
            print(f"  IMU sample {i}: t={sample.timestamp:.3f}s, "
                  f"accel={sample.accel}, gyro={sample.gyro}")
    
    # Test preintegration
    try:
        # Create a fresh preintegrator
        test_accum = PreintegratedImuMeasurements(vo.params, vo.bias0)
        
        # Preintegrate
        vo.preintegrate_between(test_accum, t_km1, t_k, 
                              imu_samples=vo.imu, 
                              imu_times=vo.imu_times,
                              R_align=R_ALIGN_IMU_TO_VO)
        
        print("Preintegration successful!")
        print(f"Delta pose: {test_accum.deltaPose().matrix()}")
        print(f"Delta velocity: {test_accum.deltaVelocity()}")
        print(f"Delta position: {test_accum.deltaPosition()}")
        
    except Exception as e:
        print(f"Preintegration failed: {e}")
def debug_isam_graph(vo, iteration=0):
    """Simplified ISAM2 debugging"""
    print(f"\n=== ISAM2 Debug - Iteration {iteration} ===")
    print(f"ISAM initialized: {vo.has_initialized_isam}")
    print(f"Current key (k): {vo.k}")
    
    if vo.has_initialized_isam:
        try:
            result = vo.isam.calculateEstimate()
            
            print("Current estimates (simplified):")
            print(f"Number of values: {result.size()}")
            
            # List all keys and their types
            keys = list(result.keys())
            print(f"Keys present: {keys}")
            
            # Try to read specific expected keys
            for i in range(vo.k + 1):  # Check poses up to current k
                pose_key = X(i)
                vel_key = V(i)
                bias_key = B(i // 5 if i > 0 else 0)  # Your bias key pattern
                
                if pose_key in keys:
                    try:
                        pose = result.atPose3(pose_key)
                        pos = pose.translation()
                        print(f"  X({i}): pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                    except Exception as e:
                        print(f"  X({i}): Error reading pose - {e}")
                
                if vel_key in keys:
                    try:
                        vel = result.atVector(vel_key)
                        print(f"  V({i}): vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
                    except:
                        pass
                
                if bias_key in keys:
                    try:
                        bias = result.atConstantBias(bias_key)
                        print(f"  B({i//5}): accel_bias=({bias.accelerometer()[0]:.3f}, {bias.accelerometer()[1]:.3f}, {bias.accelerometer()[2]:.3f})")
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error getting estimates: {e}")
    
    print(f"Graph size: {vo.graph.size()}")
    print(f"Initial values size: {vo.initial.size()}")
def debug_factor_graph_structure(vo):
    """Debug the structure of the factor graph"""
    print(f"\nFactor Graph Structure:")
    print(f"Number of factors: {vo.graph.size()}")
    
    for i in range(vo.graph.size()):
        factor = vo.graph.at(i)
        print(f"Factor {i}: {type(factor).__name__}")
        
        # Print keys this factor connects
        keys = []
        for j in range(factor.size()):
            try:
                keys.append(factor.keys()[j])
            except:
                pass
        print(f"  Connects: {keys}")












# END OF DEBUGGING IMPLEMENTATION
@dataclass
class IMUSample:
    timestamp: float
    accel: np.ndarray  # Shape (3,), in body frame [ax, ay, az]
    gyro: np.ndarray   # Shape (3,), in body frame [wx, wy, wz]
    quat_wxyz: np.ndarray | None = None  # (4,) or None if unavailable
def _estimate_bias_and_R0_from_imu(samples, t_window=1.0):
    """
    Returns:
        bias0: gtsam.imuBias.ConstantBias
        R0_np: 3x3 numpy array (body->nav) that aligns measured gravity to nav -Z
    """
    if not samples:
        return gtsam.imuBias.ConstantBias(), np.eye(3)

    t0 = samples[0].timestamp
    win = [s for s in samples if (s.timestamp - t0) <= t_window]
    if len(win) < 10:
        win = samples[:max(10, min(200, len(samples)))]

    g_meas = np.mean([s.accel for s in win], axis=0)  # body gravity + accel bias
    w_meas = np.mean([s.gyro  for s in win], axis=0)  # gyro bias (assuming still)

    # Build R0_np so that R0_np @ g_meas â‰ˆ [0, 0, -|g|]
    a = g_meas / (np.linalg.norm(g_meas) + 1e-9)
    b = np.array([0.0, 0.0, -1.0])  # nav -Z
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-8:
        R0_np = np.eye(3)
    else:
        vx = np.array([[    0, -v[2],  v[1]],
                       [  v[2],     0, -v[0]],
                       [ -v[1],  v[0],    0]], dtype=float)
        R0_np = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    # Estimate accel bias as mean specific force minus body-frame gravity implied by R0
    g_vec_nav = np.array([0.0, 0.0, -9.81])
    g_in_body = R0_np.T @ g_vec_nav
    accel_bias = g_meas - g_in_body

    bias0 = gtsam.imuBias.ConstantBias(w_meas, accel_bias)
    return bias0, R0_np

def np_Rt_to_Pose3(T:np.ndarray) -> Pose3:
    """Convert a 4x4 numpy array to a gtsam Pose3 object."""
    R = Rot3(T[:3, 0:3])
    t = Point3(T[:3, 3])
    return Pose3(R, t)

def Rx(t):
    c,s = np.cos(t), np.sin(t)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

def Rz(t):
    c,s = np.cos(t), np.sin(t)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def vector3(x,y,z):
    return np.array([x,y,z],dtype=float)
import numpy as np

def to_seconds_array(x):
    a = np.asarray(x, dtype=float).ravel()
    # Heuristic: if values look like nanoseconds, convert to seconds
    if a.size and np.nanmedian(a) > 1e12:
        a = a * 1e-9
    return a

def make_relative(t):
    t = to_seconds_array(t)
    if t.size == 0:
        return t
    return t - t[0]

def apply_relative_to_imu(imu_samples):
    if not imu_samples:
        return np.array([], float)
    t0 = float(imu_samples[0].timestamp)
    for s in imu_samples:
        s.timestamp = float(s.timestamp - t0)  # in-place
    return np.array([s.timestamp for s in imu_samples], float)

R_swap = np.array([[0,0,1],
                   [0,1,0],
                   [1,0,0]], float)     # det = -1 (mirror)
def R_yaw(theta):
    c,s = np.cos(theta), np.sin(theta)
    return np.array([[ c,0, s],
                     [ 0,1, 0],
                     [-s,0, c]], float)
CANDIDATES = [
    np.eye(3),
    np.array([[0,0,1],[0,1,0],[1,0,0]], float),      # swap x<->z
    Rx(+np.pi/2),                                     # +90° about X
    Rx(-np.pi/2),                                     # -90° about X
    np.diag([-1,1,1]),                                # flip x
    np.diag([1,1,-1]),                                # flip z
    np.array([[0,0,-1],[0,1,0],[-1,0,0]], float),     # swap + flip
    np.array([[0,0,1],[0,1,0],[1,0,0]])
    ]

def pick_best_transform(gt_xz, pred_xyz, k=50):
    return np.eye(3)
g = 9.81
n_gravity = vector3(0, 0, -g)

R_ALIGN_IMU_TO_VO = np.array([
    [0, -1, 0],   # X_m = +Y_i   (flip to -1 if your MoCap +X is robot-right)
    [0, 0, -1],   # Y_m =  Z_i
    [-1, 0, 0],   # Z_m =  -X_i
], dtype=float)
def _extract_img_index(fname: str) -> int | None:
    # Pull the integer in names like 000123.png or frame_123.jpg
    digits = ''.join(ch for ch in fname if ch.isdigit())
    return int(digits) if digits else None


def load_cam_times_from_folder(folder: str):
    """
    Reads <folder>/data.csv (timestamp, image_number) and returns a list of
    timestamps aligned with the sorted image files in <folder>.
    """
    csv_path = os.path.join(folder, "data.csv")
    t_by_num = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            try:
                t = float(row[0]); n = int(row[1])
            except ValueError:
                # skip header or bad rows
                continue
            t_by_num[n] = t

    # match timestamps to actual images on disk
    img_names = sorted([fn for fn in os.listdir(folder)
                        if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))])
    cam_times = []
    for fn in img_names:
        idx = _extract_img_index(fn)
        cam_times.append(t_by_num.get(idx, np.nan))
    return cam_times

def load_cam_times_from_zed_csv(csv_path: str, images_dir: str | None = None) -> np.ndarray:
    """
    Read 'data.csv' with columns '#timestamp [ns], frame' and return seconds.
    If images_dir is given, reorder/trim times to match PNG/JPG files present.
    """
    times_s = []
    frames  = []

    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        # simple header detection
        ts_idx = 0
        fr_idx = 1 if header and len(header) > 1 else None

        for row in r:
            if not row or len(row) <= ts_idx: 
                continue
            s = row[ts_idx].strip()
            if s.startswith("'"):  # Excel text marker
                s = s[1:].strip()
            try:
                t_ns = float(s)             # supports scientific notation
            except ValueError:
                continue
            t_s = t_ns * 1e-9               # ns → s
            times_s.append(t_s)

            if fr_idx is not None and len(row) > fr_idx:
                frames.append(os.path.basename(row[fr_idx].strip()))

    times_s = np.asarray(times_s, dtype=float)

    # If requested, align to actual images on disk using the 'frame' names
    if images_dir is not None and frames and len(frames) == len(times_s):
        # collect image basenames in sorted order (png/jpg)
        files = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                       glob.glob(os.path.join(images_dir, "*.jpg")) +
                       glob.glob(os.path.join(images_dir, "*.jpeg")))

        basenames = [os.path.basename(p) for p in files]
        # map frame name -> time
        name2t = {fn: t for fn, t in zip(frames, times_s)}
        # build time array in the same order as images; drop missing
        t_aligned = np.array([name2t.get(fn, np.nan) for fn in basenames], float)
        mask = np.isfinite(t_aligned)
        return t_aligned[mask]  # times aligned to images_dir

    # otherwise just return the raw times
    return times_s

def _to_seconds(ts):
    """
    Convert a numeric timestamp to seconds.
    Heuristic: if ts is large magnitude, assume ns/us/ms and convert.
    """
    ts = float(ts)
    if ts > 1e14:         # ~ > 3 years in ns
        return ts * 1e-9  # ns -> s
    elif ts > 1e11:       # microseconds
        return ts * 1e-6
    elif ts > 1e8:        # milliseconds
        return ts * 1e-3
    else:
        return ts         # assume seconds already

def _quat_from_ypr(yaw, pitch, roll):
    """Build quaternion (w,x,y,z) from yaw(Z), pitch(Y), roll(X) using ZYX convention."""
    cy = math.cos(yaw * 0.5);  sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    w = cr*cp*cy + sr*cp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w,x,y,z], float)

def _rot3_from_wxyz(q):
    # q: (w,x,y,z), normalized
    w,x,y,z = q
    return gtsam.Rot3.Quaternion(w, x, y, z)

def _first_imu_rot_world_to_cam(self):
    """
    Returns Rot3 world->cam from the first IMU quaternion, or None if unavailable.
    Assumes:
      - self.imu_samples: list of IMUSample with .quat_wxyz=(w,x,y,z) or None
      - self.imu_quat_mapping: "world_to_imu" or "imu_to_world"   (default: "world_to_imu")
      - self.imu_world_frame:  "ENU" or "NED"                      (default: "ENU")
      - self.T_ic: Pose3 IMU->Camera extrinsic (optional; uses identity if missing)
    """
    if not hasattr(self, "imu") or not self.imu:
        return None
    idx0 = next((i for i,s in enumerate(self.imus) if getattr(s, "quat_wxyz", None) is not None), None)
    if idx0 is None:
        return None

    q_wxyz = self.imu[idx0].quat_wxyz
    # world->imu as default
    R_wi = _rot3_from_wxyz(q_wxyz)
    if getattr(self, "imu_quat_mapping", "world_to_imu") == "imu_to_world":
        R_wi = R_wi.inverse()

    # If your IMU quats are NED, convert world (NED) to ENU
    if getattr(self, "imu_world_frame", "ENU").upper() == "NED":
        # NED->ENU rotation: swap x<->y, flip z
        R_ned_to_enu = gtsam.Rot3(np.array([[0,1,0],[1,0,0],[0,0,-1]], float))
        R_wi = R_ned_to_enu.compose(R_wi)

    # IMU->Cam extrinsic
    T_ic = getattr(self, "T_ic", Pose3())  # Pose3 identity if absent

    # world->cam = world->imu ∘ imu->cam
    R_wc = R_wi.compose(T_ic.rotation())
    return R_wc
def load_imu_csv_ns(csv_path, has_header=True):
    """
    Expected columns (by name OR position):
      timestamp(ns), orientation_x, orientation_y, orientation_z, [orientation_w optional],
      angular_velocity_x, angular_velocity_y, angular_velocity_z,
      linear_acceleration_x, linear_acceleration_y, linear_acceleration_z

    Returns: List[IMUSample] with timestamp in seconds and quat_wxyz if available.
    """
    samples = []
    with open(csv_path, "r", newline="") as f:
        if has_header:
            r = csv.DictReader(f)

            # Flexible key lookup (case/space insensitive)
            def g(row, *names, default=None, required=False):
                for n in names:
                    if n in row and row[n] != "":
                        return row[n]
                low = {k.strip().lower(): v for k, v in row.items()}
                for n in names:
                    n2 = n.strip().lower()
                    if n2 in low and low[n2] != "":
                        return low[n2]
                if required:
                    raise KeyError(names)
                return default

            for row in r:
                try:
                    t  = _to_seconds(g(row, "#timestamp [ns]","timestamp","time","t", required=True))
                    gx = float(g(row, "angular_velocity_x","gyro_x","wx","gyr_x", required=True))
                    gy = float(g(row, "angular_velocity_y","gyro_y","wy","gyr_y", required=True))
                    gz = float(g(row, "angular_velocity_z","gyro_z","wz","gyr_z", required=True))
                    ax = float(g(row, "linear_acceleration_x","accel_x","ax","accl_x", required=True))
                    ay = float(g(row, "linear_acceleration_y","accel_y","ay","accl_y", required=True))
                    az = float(g(row, "linear_acceleration_z","accel_z","az","accl_z", required=True))

                    # Orientation candidates
                    ox = g(row, "orientation_x","qx","q_x","ori_x")
                    oy = g(row, "orientation_y","qy","q_y","ori_y")
                    oz = g(row, "orientation_z","qz","q_z","ori_z")
                    ow = g(row, "orientation_w","qw","q_w","w")

                    quat = None
                    if ox is not None and oy is not None and oz is not None and ow is not None:
                        # Full quaternion present
                        q = np.array([float(ow), float(ox), float(oy), float(oz)], float)
                        nrm = np.linalg.norm(q)
                        if nrm > 0:
                            q /= nrm
                            if q[0] < 0:  # force w >= 0 for a consistent hemisphere
                                q *= -1.0
                            quat = q
                    elif ox is not None and oy is not None and oz is not None:
                        # Try to recover w from unit norm
                        x = float(ox); y = float(oy); z = float(oz)
                        s = 1.0 - (x*x + y*y + z*z)
                        if s >= 0.0:
                            w = math.sqrt(s)
                            q = np.array([w,x,y,z], float)
                            q /= np.linalg.norm(q)
                            if q[0] < 0:
                                q *= -1.0
                            quat = q
                        else:
                            # Try yaw/pitch/roll fallback
                            ycol = g(row, "yaw","psi","heading")
                            pcol = g(row, "pitch","theta")
                            rcol = g(row, "roll","phi")
                            if ycol is not None and pcol is not None and rcol is not None:
                                quat = _quat_from_ypr(float(ycol), float(pcol), float(rcol))
                                if quat[0] < 0:
                                    quat *= -1.0
                    else:
                        # Maybe we only have ypr
                        ycol = g(row, "yaw","psi","heading")
                        pcol = g(row, "pitch","theta")
                        rcol = g(row, "roll","phi")
                        if ycol is not None and pcol is not None and rcol is not None:
                            quat = _quat_from_ypr(float(ycol), float(pcol), float(rcol))
                            if quat[0] < 0:
                                quat *= -1.0

                except Exception:
                    continue  # skip malformed

                samples.append(IMUSample(
                    timestamp=t,
                    accel=np.array([ax, ay, az], float),
                    gyro =np.array([gx, gy, gz], float),
                    quat_wxyz=quat
                ))

        else:
            # Fixed-pos fallback:
            # 0: timestamp(ns) | 1-3: orientation_x/y/z | 4-6: angular vel | 7-9: linear accel | [10]=orientation_w optional
            r = csv.reader(f)
            for row in r:
                if not row or len(row) < 10:
                    continue
                try:
                    t  = _to_seconds(row[0])
                    gx = float(row[4]); gy = float(row[5]); gz = float(row[6])
                    ax = float(row[7]); ay = float(row[8]); az = float(row[9])
                    quat = None
                    if len(row) >= 11:
                        ox = float(row[1]); oy = float(row[2]); oz = float(row[3]); ow = float(row[10])
                        q = np.array([ow, ox, oy, oz], float)
                        q /= np.linalg.norm(q) if np.linalg.norm(q) > 0 else 1.0
                        if q[0] < 0: q *= -1.0
                        quat = q
                    else:
                        # try unit-norm recovery
                        ox = float(row[1]); oy = float(row[2]); oz = float(row[3])
                        s = 1.0 - (ox*ox + oy*oy + oz*oz)
                        if s >= 0:
                            ow = math.sqrt(s)
                            q = np.array([ow, ox, oy, oz], float)
                            q /= np.linalg.norm(q)
                            if q[0] < 0: q *= -1.0
                            quat = q
                except Exception:
                    continue
                samples.append(IMUSample(
                    timestamp=t,
                    accel=np.array([ax, ay, az], float),
                    gyro =np.array([gx, gy, gz], float),
                    quat_wxyz=quat
                ))
    return samples

def _interp_vec(tq, t0, v0, t1, v1):
    """Linear interpolate vector v(t) at tq between (t0,v0) and (t1,v1)."""
    if tq <= t0: return v0
    if tq >= t1: return v1
    w = (tq - t0) / max(1e-9, (t1 - t0))
    return (1.0 - w) * v0 + w * v1

def _interp_imu_at(time_s, imu_samples, imu_times):
    """
    Return (a, w) linearly interpolated at 'time_s'.
    If time outside range, clamp to nearest sample.
    """
    if len(imu_samples) == 0:
        raise ValueError("No IMU samples loaded")
    idx = np.searchsorted(imu_times, time_s, side="left")

    if idx <= 0:
        s0 = imu_samples[0]
        return s0.accel.astype(float), s0.gyro.astype(float)
    if idx >= len(imu_samples):
        s1 = imu_samples[-1]
        return s1.accel.astype(float), s1.gyro.astype(float)

    s0 = imu_samples[idx-1]; s1 = imu_samples[idx]
    t0 = imu_times[idx-1];   t1 = imu_times[idx]
    a = _interp_vec(time_s, t0, s0.accel, t1, s1.accel).astype(float)
    w = _interp_vec(time_s, t0, s0.gyro,  t1, s1.gyro ).astype(float)
    return a, w
class VisualOdometry():
    def __init__(self,
             calib_path: str,
             left_dir: str,
             right_dir: str,
             poses_path: str | None = None,
             gt_poses: List[np.ndarray] | None = None,
             imu_csv: str | None = None):
        """
        calib_path: absolute path to calib.txt
        left_dir:   folder with left images AND data.csv
        right_dir:  folder with right images AND data.csv
        poses_path: optional ground-truth poses (KITTI style), else None
        imu_csv:    optional IMU CSV path; if provided we load it here
        """
        # -------- Paths ----------
        self.calib_path = calib_path
        self.left_dir   = left_dir
        self.right_dir  = right_dir

        # -------- Calibration ----------
        self.K_l, self.P_l, self.K_r, self.P_r = self._load_calib(self.calib_path)

        # -------- Optional GT ----------
        if gt_poses is not None and len(gt_poses) > 0:
            self.gt_poses = gt_poses
        else:
            self.gt_poses = self._load_poses(poses_path) if poses_path else []

        # -------- Images & timestamps ----------
        self.images_l = self._load_images(self.left_dir)
        self.images_r = self._load_images(self.right_dir)

        # Left timeline (primary)
        self.cam_times   = load_cam_times_from_zed_csv(LEFT_DATACSV, images_dir=self.left_dir)
        # Right timeline (not used elsewhere, but we keep it for diagnostics)
        self.cam_times_r = load_cam_times_from_zed_csv(RIGHT_DATACSV, images_dir=self.right_dir)

        # -------- Stereo disparity ----------
        block = 11
        P1 = block * block * 8
        P2 = block * block * 32
        self.disparity = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=32, blockSize=block, P1=P1, P2=P2
        )
        if len(self.images_l) > 0 and len(self.images_r) > 0:
            first_disp = self.disparity.compute(self.images_l[0], self.images_r[0]).astype(np.float32) / 16.0
        else:
            first_disp = np.zeros((1, 1), dtype=np.float32)
        self.disparities = [first_disp]

        # -------- Front-ends ----------
        self.fastFeatures = cv2.FastFeatureDetector_create()
        self.lk_params = dict(
            winSize=(15, 15),
            flags=cv2.MOTION_AFFINE,
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03)
        )

        # -------- IMU / preintegration (names consistent with fuse_vo_imu_step) ----------
        # GTSAM preintegration params (world Z up)
        I3 = np.eye(3)
        self.params = PreintegrationParams.MakeSharedU(9.81)
        self.params.setAccelerometerCovariance(I3 * (0.02**2))                 # tune as needed
        self.params.setGyroscopeCovariance(I3 * (np.deg2rad(0.5)**2))           # tune as needed
        self.params.setIntegrationCovariance(I3 * (1e-4))                       # tune as needed

        # Initial bias + preintegrator
        self.bias0 = imuBias.ConstantBias()
        self.accum = PreintegratedImuMeasurements(self.params, self.bias0)

        # IMU samples container (use the same name your helpers expect: self.imu)
        self.imu: List[IMUSample] = []
        self.imu_times = np.array([], dtype=float)
        if imu_csv:
            try:
                self.imu = load_imu_csv_ns(imu_csv, has_header=True)
                self.imu_times = np.array([s.timestamp for s in self.imu], dtype=float)
                print(f"[IMU] loaded {len(self.imu)} samples from {imu_csv}")
            except Exception as e:
                print(f"[IMU] failed to load {imu_csv}: {e}")

        # Alignment of raw IMU samples -> VO/body frame (your fuse uses global R_ALIGN_IMU_TO_VO)
        # Keep using the global R_ALIGN_IMU_TO_VO already defined in this file.

        # -------- Extrinsics (Camera <-> IMU) ----------
        # Your fuse uses self.T_cam_imu (Camera->IMU). Keep that name & API.
        self.T_cam_imu = Pose3()  # override later via set_cam_imu_extrinsics(...)
        # (You also have T_ic uses elsewhere; keep ONLY T_cam_imu to avoid confusion.)

        # -------- Factor graph & solver ----------
        self.graph   = NonlinearFactorGraph()
        self.initial = Values()
        self.isam    = ISAM2()

        # -------- State / keys / flags ----------
        self.k = 0
        self.has_initialized_isam = False
        self.bias_key    = B(0)   # matches _initialise_isam / fuse usage
        self.cur_vel_key = V(0)

        # -------- Noise models (single source of truth) ----------
        self.pose_prior_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3], dtype=float)  # rpy(rad), xyz(m)
        )
        self.vel_prior_noise  = noiseModel.Isotropic.Sigma(3, 0.05)
        self.bias_prior_noise = noiseModel.Isotropic.Sigma(6, 0.05)
        self.odo_noise        = noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10], dtype=float)
        )

        # Units-check flag for IMU (used in fuse_vo_imu_step)
        self._imu_units_checked = False

    def preintegrate_between(preint, t0, t1, imu_samples, imu_times, R_align=np.eye(3)):
        """
        Reset 'preint' and integrate IMU from t0 to t1 using:
        - interpolated (a,w) at t0,
        - all samples in (t0, t1],
        - final segment up to t1 with last (a,w).
        No global time shift; exact camera times are used as boundaries.
        """
        assert t1 > t0, "t1 must be > t0"
        preint.resetIntegration()

        # Endpoint at t0
        a_prev, w_prev = _interp_imu_at(t0, imu_samples, imu_times)
        a_prev = (R_align @ a_prev).astype(float)
        w_prev = (R_align @ w_prev).astype(float)
        t_prev = t0

        # All IMU samples with t in (t0, t1]
        lo = np.searchsorted(imu_times, t0, side="right")
        hi = np.searchsorted(imu_times, t1, side="right")

        for k in range(lo, hi):
            s = imu_samples[k]
            tk = s.timestamp
            dt = max(1e-6, tk - t_prev)
            preint.integrateMeasurement(a_prev, w_prev, dt)
            # step to the current sample (piecewise-constant)
            a_prev = (R_align @ s.accel).astype(float)
            w_prev = (R_align @ s.gyro ).astype(float)
            t_prev = tk

        # Final segment to t1 using last held (a_prev, w_prev)
        if t_prev < t1 - 1e-12:
            dt = max(1e-6, t1 - t_prev)
            preint.integrateMeasurement(a_prev, w_prev, dt)
    
    def _initialize_isam(self, pose0: Pose3):
        # Pose prior at k=0
        self.graph.add(PriorFactorPose3(X(0), pose0, self.pose_prior_noise))
        # Velocity prior at k=0
        self.graph.add(PriorFactorVector(V(0), np.zeros(3), self.vel_prior_noise))
        self.initial.insert(V(0), np.zeros(3))
        # Bias prior at B(0)
        self.graph.add(PriorFactorConstantBias(B(0), self.bias0, self.bias_prior_noise))
        self.initial.insert(B(0), self.bias0)

        # Initial pose value
        self.initial.insert(X(0), pose0)

        # First update
        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial.clear()
        self.has_initialized_isam = True

    def _imu_window(self, t0, t1):
        return [s for s in self.imu if (s.timestamp > t0 and s.timestamp <= t1)]

    def fuse_vo_imu_step(self, T_km1_k_4x4: np.ndarray, t_km1: float | None = None, t_k: float | None = None) -> Pose3:
        # Initialize once
        # Add this at the very beginning:
        print(f"\n=== FUSE VO IMU STEP k={self.k+1} ===")
        print(f"VO translation: {T_km1_k_4x4[:3, 3]}")
        print(f"VO translation magnitude: {np.linalg.norm(T_km1_k_4x4[:3, 3]):.6f}")
        if not self.has_initialized_isam:
            pose0 = Pose3()  # or a Pose3 built from your gravity-aligned R0 if you have it
            self._initialize_isam(pose0)

        k = self.k + 1

        # Convert VO delta (camera frame) to a Pose3
        R_cam = Rot3(T_km1_k_4x4[:3, :3]); t_cam = Point3(*T_km1_k_4x4[:3, 3])
        T_meas_cam = Pose3(R_cam, t_cam)

        # Conjugate to IMU/body frame: T^imu = (T_cam_imu)^{-1} * T^cam * (T_cam_imu)
        T_meas_imu = self.T_cam_imu.inverse().compose(T_meas_cam).compose(self.T_cam_imu)

        

        odo_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10])
        )

        # ---- Add IMU factor if we have samples in (t_{k-1}, t_k] ----
        added_imu = False
        if (t_km1 is not None) and (t_k is not None) and (len(self.imu) > 0):
            # Optional one-time units sanity (deg/s -> rad/s) using a short window
            if not getattr(self, "_imu_units_checked", False):
                lo = np.searchsorted(self.imu_times, t_km1, side="right")
                hi = np.searchsorted(self.imu_times, t_k,   side="right")
                window = self.imu[lo:hi]
                if len(window) >= 10:
                    mags = [np.linalg.norm(s.gyro) for s in window[:min(200, len(window))]]
                    if np.median(mags) > np.deg2rad(200):
                        for s in self.imu:
                            s.gyro = np.deg2rad(s.gyro)
                self._imu_units_checked = True

            # Preintegrate strictly between camera times (t_{k-1}, t_k]
            self.preintegrate_between(self.accum, t_km1, t_k,
                                imu_samples=self.imu,
                                imu_times=self.imu_times,
                                R_align=R_ALIGN_IMU_TO_VO)  # or self.R_align_imu_to_body

            self.graph.add(ImuFactor(X(k-1), V(k-1), X(k), V(k), self.bias_key, self.accum))
            added_imu = True

        # Optional: bias chaining
        if added_imu and (k % 5 == 0):
            new_bias_key = B(k // 5)
            self.graph.add(BetweenFactorConstantBias(
                self.bias_key, new_bias_key,
                gtsam.imuBias.ConstantBias(),
                noiseModel.Isotropic.Variance(6, 0.1)
            ))
            self.initial.insert(new_bias_key, gtsam.imuBias.ConstantBias())
            self.bias_key = new_bias_key

        # Initial guesses for new states
        prev_pose = self.isam.calculateEstimate().atPose3(X(k - 1))
        self.initial.insert(X(k), prev_pose.compose(T_meas_imu))
        self.initial.insert(V(k), np.zeros(3))

        # Update & return current pose
        self.isam.update(self.graph, self.initial)
        result = self.isam.calculateEstimate()
        self.graph = NonlinearFactorGraph()
        self.initial.clear()

        self.k = k
        return result.atPose3(X(k))


    def _pose3_from_T(self, T_4x4: np.ndarray) -> Pose3:
        R = Rot3(T_4x4[:3, :3])
        t = Point3(*T_4x4[:3, 3])
        return Pose3(R, t)

    def initialize_isam_with_prior(self, T0_4x4: np.ndarray | None = None):
        """One-time: add PriorFactorPose3 at X(0), insert initial, and do first isam.update()."""
        if self.has_initialized_isam:
            return

        # ALWAYS use GT initial pose if available
        if hasattr(self, 'gt_poses') and len(self.gt_poses) > 0:
            # Use the first GT pose as starting position
            T0_4x4 = self.gt_poses[0]
            print(f"Using GT initial pose: position ({T0_4x4[0,3]:.3f}, {T0_4x4[1,3]:.3f}, {T0_4x4[2,3]:.3f})")
        
        if T0_4x4 is not None:
            pose0 = self._pose3_from_T(T0_4x4)
        else:
            # Try IMU-based attitude init but keep position at GT start
            R_wc0 = _first_imu_rot_world_to_cam(self)
            if R_wc0 is not None:
                # Start at GT position with IMU-derived attitude
                if hasattr(self, 'gt_poses') and len(self.gt_poses) > 0:
                    gt_pos = self.gt_poses[0][:3, 3]
                    pose0 = Pose3(R_wc0, gtsam.Point3(*gt_pos))
                    print(f"Using GT position with IMU attitude: {gt_pos}")
                else:
                    pose0 = Pose3(R_wc0, gtsam.Point3(0.0, 0.0, 0.0))
            else:
                # Fallback to GT position with identity rotation
                if hasattr(self, 'gt_poses') and len(self.gt_poses) > 0:
                    gt_pos = self.gt_poses[0][:3, 3]
                    pose0 = Pose3(Rot3(), gtsam.Point3(*gt_pos))
                    print(f"Using GT position with identity rotation: {gt_pos}")
                else:
                    pose0 = Pose3()

        # Prior factor at X(0)
        self.graph.add(PriorFactorPose3(X(0), pose0, self.pose_prior_noise))
        # Initial guess for X(0)
        self.initial.insert(X(0), pose0)

        # First iSAM update
        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial.clear()
        self.k = 0
        self.has_initialized_isam = True


    def add_vo_between_and_update(self, T_km1_k_4x4: np.ndarray):
        """
        Add a VO BetweenFactorPose3 between X(k-1) and X(k), provide initial for X(k),
        then run a single iSAM2 update. Returns the current Pose3 at X(k).
        """
        # Next index
        k = self.k + 1

        # Measurement as Pose3
        T_meas = self._pose3_from_T(T_km1_k_4x4)

        # Between factor: X(k-1) --T_meas--> X(k)
        self.graph.add(BetweenFactorPose3(X(k - 1), X(k), T_meas, self.odo_noise))

        # Initial guess for X(k): compose last estimate with VO delta
        prev_pose = self.isam.calculateEstimate().atPose3(X(k - 1))
        self.initial.insert(X(k), prev_pose.compose(T_meas))

        # Incremental solve
        self.isam.update(self.graph, self.initial)
        result = self.isam.calculateEstimate()

        # Reset temporary containers
        self.graph = NonlinearFactorGraph()
        self.initial.clear()

        # Advance index
        self.k = k
        return result.atPose3(X(k))
    
    def set_cam_imu_extrinsics(self, R_cam_imu: np.ndarray | None = None, t_cam_imu: np.ndarray | None = None):
        if R_cam_imu is None: R_cam_imu = np.eye(3)
        if t_cam_imu is None: t_cam_imu = np.zeros(3)
        self.T_cam_imu = self._pose_from_R_t(R_cam_imu, t_cam_imu)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera from a KITTI-style calib.txt:
        First line: 12 numbers (3x4) for left; second line: 12 numbers (3x4) for right.
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_l = np.reshape(params, (3, 4))
            K_l = P_l[0:3, 0:3]
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P_r = np.reshape(params, (3, 4))
            K_r = P_r[0:3, 0:3]
        return K_l, P_l, K_r, P_r

    @staticmethod
    def _load_poses(filepath):
        """
        Loads GT poses if the file exists; otherwise returns [].
        """
        if filepath is None or (not os.path.exists(filepath)):
            return []
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads grayscale images from a directory, filtering out non-image files.
        """
        if not os.path.isdir(filepath):
            raise FileNotFoundError(f"Image directory not found: {filepath}")
        image_files = [fn for fn in sorted(os.listdir(filepath))
                       if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))]
        image_paths = [os.path.join(filepath, fn) for fn in image_files]
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        # Drop any Nones defensively
        images = [im for im in images if im is not None]
        return images

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    def _pose_from_R_t(self, R: np.ndarray, t: np.ndarray) -> Pose3:
        return Pose3(Rot3(R), Point3(*t))

    def _conj(self, T_parent_child: Pose3, T_child_delta: Pose3) -> Pose3:
        # return T_parent_delta = T_parent_child * T_child_delta * T_parent_child^{-1}
        return T_parent_child.compose(T_child_delta).compose(T_parent_child.inverse())

    def reprojection_residuals(self, dof, q1, q2, Q1, Q2):
        """
        Calculate residuals for LM refinement
        """
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

    def get_tiled_keypoints(self, img, tile_h, tile_w):
        """
        Splits the image into tiles and detects the 10 best keypoints in each tile
        """
        def get_kps(x, y):
            impatch = img[y:y + tile_h, x:x + tile_w]
            keypoints = self.fastFeatures.detect(impatch)
            for pt in keypoints:
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
            if len(keypoints) > 10:
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints

        h, w, *_ = img.shape
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]
        if len(kp_list) == 0:
            return np.empty((0,), dtype=object)
        kp_list_flatten = np.concatenate(kp_list) if len(kp_list) > 1 else np.array(kp_list[0])
        return kp_list_flatten

    def track_keypoints(self, img1, img2, kp1, max_error=4):
        """
        Tracks the keypoints between frames
        """
        if len(kp1) == 0:
            return np.empty((0,2)), np.empty((0,2))

        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)

        trackable = st.astype(bool)
        under_thresh = np.where(err[trackable] < max_error, True, False)

        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        h, w = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]

        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1, q2, disp1, disp2, min_disp=0.0, max_disp=100.0):
        """
        Calculates the right keypoints (feature points)
        """
        def get_idxs(q, disp):
            q_idx = q.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)

        in_bounds = np.logical_and(mask1, mask2)

        q1_l, q2_l, disp1, disp2 = q1[in_bounds], q2[in_bounds], disp1[in_bounds], disp2[in_bounds]

        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2

        return q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l, q1_r, q2_l, q2_r):
        """
        Triangulate points from both images
        """
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)
        Q1 = np.transpose(Q1[:3] / Q1[3])

        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)
        Q2 = np.transpose(Q2[:3] / Q2[3])
        return Q1, Q2

    def estimate_pose(self, q1, q2, Q1, Q2, max_iter=100):
        """
        Estimates the transformation matrix
        """
        early_termination_threshold = 5
        min_error = float('inf')
        early_termination = 0

        if q1.shape[0] < 6:
            return np.eye(4)

        for _ in range(max_iter):
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            in_guess = np.zeros(6)
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

        r = out_pose[:3]
        R, _ = cv2.Rodrigues(r)
        t = out_pose[3:]
        transformation_matrix = self._form_transf(R, t)
        return transformation_matrix

    def get_pose(self, i):
        """
        Calculates the transformation matrix for the i'th frame
        """
        img1_l, img2_l = self.images_l[i - 1:i + 1]

        kp1_l = self.get_tiled_keypoints(img1_l, 10, 20)
        tp1_l, tp2_l = self.track_keypoints(img1_l, img2_l, kp1_l)
        if tp1_l.shape[0] < 6:
            return np.eye(4)

        disp_curr = self.disparity.compute(img2_l, self.images_r[i]).astype(np.float32) / 16.0
        self.disparities.append(disp_curr)

        tp1_l, tp1_r, tp2_l, tp2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[i - 1], self.disparities[i])
        if tp1_l.shape[0] < 6:
            return np.eye(4)

        Q1, Q2 = self.calc_3d(tp1_l, tp1_r, tp2_l, tp2_r)
        transformation_matrix = self.estimate_pose(tp1_l, tp2_l, Q1, Q2)
        return transformation_matrix

    def attach_imu(self, imu_samples: List[IMUSample], cam_times: List[float]):
        """
        Call this once before main loop.
        - imu_samples: high-rate IMU in body frame.
        - cam_times: time stamps per left image (seconds, same epoch as IMU)
        """
        self.imu_samples = imu_samples
        self.cam_times = cam_times

    def _initialise_isam(self, pose0: Pose3):
        # Pose prior
        self.graph.add(PriorFactorPose3(X(0), pose0, self.pose_prior_noise))
        # Bias Prior
        self.graph.add(PriorFactorConstantBias(self.bias_key, gtsam.imuBias.ConstantBias(), self.bias_prior_noise))
        self.initial.insert(self.bias_key, gtsam.imuBias.ConstantBias())
        # Velocity prior (start at zero or small value)
        self.graph.add(PriorFactorVector(self.cur_vel_key, np.zeros(3), self.vel_prior_noise))
        self.initial.insert(self.cur_vel_key, np.zeros(3))
        # Also seed pose0
        self.initial.insert(X(0), pose0)

        self.isam.update(self.graph, self.initial)
        self.graph = NonlinearFactorGraph()
        self.initial.clear()

        self.has_initialized_isam = True


class ORBBackend:
    def __init__(self,
                 nfeatures=2000,
                 scaleFactor=1.2,
                 nlevels=8,
                 edgeThreshold=31,
                 WTA_K=2,
                 scoreType=cv2.ORB_HARRIS_SCORE,
                 patchSize=31,
                 fastThreshold=20,
                 ratio=0.75):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            WTA_K=WTA_K,
            scoreType=scoreType,
            patchSize=patchSize,
            fastThreshold=fastThreshold
        )
        self.ratio = ratio
        # Hamming for ORBâ€™s binary descriptors
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_describe(self, img_gray):
        """Detect keypoints & ORB descriptors. Returns (Nx2 float32 points, Nx32 uint8 descriptors)."""
        kps, des = self.orb.detectAndCompute(img_gray, None)
        if des is None or len(kps) == 0:
            return np.empty((0, 2), dtype=np.float32), None
        pts = np.asarray([k.pt for k in kps], dtype=np.float32)
        return pts, des

    def match(self, des1, des2):
        """
        KNN match + Loweâ€™s ratio test.
        Returns arrays of indices into pts1/pts2 that correspond to 'good' matches.
        """
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), []

        knn = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < self.ratio * n.distance:
                good.append(m)

        idx1 = np.array([m.queryIdx for m in good], dtype=np.int32)
        idx2 = np.array([m.trainIdx for m in good], dtype=np.int32)
        return idx1, idx2, good


class TimeSync:
    def __init__(self, imu_path, cam_path, mocap_path,
                 mocap_rot_in_radians=True,
                 grid_dt=0.01,
                 tau_search=10.0,
                 cam_imu_search=(-10.0, 10.0, 0.005)):
        """
        imu_path   : CSV file OR directory containing IMU CSV
        cam_path   : CSV file OR directory; if directory, tries <dir>/data.csv or first *.csv
        mocap_path : CSV file OR directory containing MoCap CSV
        """
        self.imu_path = self._resolve_csv(imu_path)
        self.cam_path = self._resolve_csv(cam_path, prefer_name="data.csv")
        self.mocap_path = self._resolve_csv(mocap_path)
        self.mocap_rot_in_radians = mocap_rot_in_radians
        self.grid_dt = float(grid_dt)
        self.tau_search = float(tau_search)
        self.cam_imu_search = cam_imu_search

        # Filled after run()
        self.synced = None             # pandas.DataFrame
        self.tau_add_to_mocap = None   # add to MoCap to align to IMU (s)
        self.tau_add_to_cam = None     # add to Camera to align to IMU (s)
        self.mean_abs_err_cam_imu = None

    # ---------- public ----------
    def run(self):
        # Load streams
        t_imu, w_imu, a_imu, imu_df = self._load_imu(self.imu_path)
        t_cam, frame_idx, cam_df    = self._load_cam_times(self.cam_path)
        t_mocap, rot_m, pos_m       = self._load_mocap(self.mocap_path, radians=self.mocap_rot_in_radians)

        # 1) IMU<->MoCap offset via omega-based cross-correlation
        tau_imu_to_align = self._estimate_offset_omega(t_mocap, rot_m, t_imu, w_imu,
                                                       grid_dt=self.grid_dt, tau_search=self.tau_search)
        # We want to ADD a shift to MoCap to land on IMU time
        self.tau_add_to_mocap = -tau_imu_to_align
        t_mocap_aligned = t_mocap + self.tau_add_to_mocap

        # 2) Camera<->IMU constant offset via nearest-neighbor time only
        self.tau_add_to_cam, self.mean_abs_err_cam_imu = self._estimate_offset_by_nearest(
            t_ref=t_imu, t_mov=t_cam, search=self.cam_imu_search)
        t_cam_aligned = t_cam + self.tau_add_to_cam

        # 3) Interpolate IMU & MoCap to each camera time
        synced = pd.DataFrame({
            "frame": frame_idx[:len(t_cam_aligned)],
            "t_cam_from_start_s": t_cam[:len(t_cam_aligned)],
            "t_cam_aligned_to_imu_s": t_cam_aligned[:len(t_cam_aligned)],
        })

        if w_imu is not None:
            synced["imu_wx"] = np.interp(t_cam_aligned, t_imu, w_imu[:,0])
            synced["imu_wy"] = np.interp(t_cam_aligned, t_imu, w_imu[:,1])
            synced["imu_wz"] = np.interp(t_cam_aligned, t_imu, w_imu[:,2])
        if a_imu is not None:
            synced["imu_ax"] = np.interp(t_cam_aligned, t_imu, a_imu[:,0])
            synced["imu_ay"] = np.interp(t_cam_aligned, t_imu, a_imu[:,1])
            synced["imu_az"] = np.interp(t_cam_aligned, t_imu, a_imu[:,2])

        # MoCap (rot+pos)
        synced["mocap_rot_x"] = np.interp(t_cam_aligned, t_mocap_aligned, rot_m[:,0])
        synced["mocap_rot_y"] = np.interp(t_cam_aligned, t_mocap_aligned, rot_m[:,1])
        synced["mocap_rot_z"] = np.interp(t_cam_aligned, t_mocap_aligned, rot_m[:,2])
        synced["mocap_pos_x"] = np.interp(t_cam_aligned, t_mocap_aligned, pos_m[:,0])
        synced["mocap_pos_y"] = np.interp(t_cam_aligned, t_mocap_aligned, pos_m[:,1])
        synced["mocap_pos_z"] = np.interp(t_cam_aligned, t_mocap_aligned, pos_m[:,2])

        self.synced = synced
        return synced

    # ---------- loaders ----------
    def _resolve_csv(self, path, prefer_name=None):
        path = os.path.normpath(path)
        if os.path.isdir(path):
            if prefer_name and os.path.exists(os.path.join(path, prefer_name)):
                return os.path.join(path, prefer_name)
            cands = sorted(glob.glob(os.path.join(path, "*.csv")))
            if not cands:
                raise FileNotFoundError(f"No CSV files found in directory: {path}")
            return cands[0]
        else:
            if not os.path.exists(path):
                if os.path.exists(path + ".csv"):
                    return path + ".csv"
                raise FileNotFoundError(f"File not found: {path}")
            return path

    @staticmethod
    def _to_seconds(ts):
        ts = float(ts)
        if ts > 1e14:   # ns
            return ts * 1e-9
        elif ts > 1e11: # us
            return ts * 1e-6
        elif ts > 1e8:  # ms
            return ts * 1e-3
        else:
            return ts

    def _load_imu(self, path):
        df = pd.read_csv(path)
        ts_col = df.columns[0]
        t_raw = pd.to_numeric(df[ts_col], errors="coerce").to_numpy()
        t_s   = np.array([self._to_seconds(x) for x in t_raw], dtype=float)
        t_s   = t_s - t_s[0]

        def find(aliases):
            cols = list(df.columns)
            low  = {c.lower(): c for c in cols}
            for name in aliases:
                if name.lower() in low:
                    return low[name.lower()]
            for c in cols:
                if any(name.lower() in c.lower() for name in aliases):
                    return c
            return None

        gx = find(["gyro_x","angular_velocity_x","wx","omega_x"])
        gy = find(["gyro_y","angular_velocity_y","wy","omega_y"])
        gz = find(["gyro_z","angular_velocity_z","wz","omega_z"])
        ax = find(["linear_acceleration_x","accel_x","ax","a_x"])
        ay = find(["linear_acceleration_y","accel_y","ay","a_y"])
        az = find(["linear_acceleration_z","accel_z","az","a_z"])

        w = None; a = None
        if gx and gy and gz:
            w = np.vstack([
                pd.to_numeric(df[gx], errors="coerce").to_numpy(),
                pd.to_numeric(df[gy], errors="coerce").to_numpy(),
                pd.to_numeric(df[gz], errors="coerce").to_numpy(),
            ]).T
        if ax and ay and az:
            a = np.vstack([
                pd.to_numeric(df[ax], errors="coerce").to_numpy(),
                pd.to_numeric(df[ay], errors="coerce").to_numpy(),
                pd.to_numeric(df[az], errors="coerce").to_numpy(),
            ]).T
        return t_s, w, a, df

    def _load_cam_times(self, path):
        df = pd.read_csv(path, header=None)
        try:
            float(df.iloc[0,0])
        except Exception:
            df = pd.read_csv(path)

        candidates = {c.lower(): c for c in df.columns}
        ts_col = None; idx_col = None
        for key in ["timestamp","time","t"]:
            if key in candidates: ts_col = candidates[key]; break
        for key in ["image_number","frame","index","id","img","image"]:
            if key in candidates: idx_col = candidates[key]; break

        if ts_col is None: ts_col = df.columns[0]
        if idx_col is None: idx_col = df.columns[1] if len(df.columns) > 1 else None

        t_raw = pd.to_numeric(df[ts_col], errors="coerce").dropna().to_numpy()
        t_s   = np.array([self._to_seconds(x) for x in t_raw], dtype=float)
        t_s   = t_s - t_s[0]
        frame = (pd.to_numeric(df[idx_col], errors="coerce").fillna(-1).astype(int).to_numpy()
                 if idx_col is not None else np.arange(len(t_s)))
        return t_s, frame, df

    def _load_mocap(self, path, radians=True):
        df = pd.read_csv(path)
        # time
        time_col = self._find_col(df, ["time","t","sec","seconds"])
        if time_col is None:
            t = np.arange(len(df)) * self.grid_dt
        else:
            t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
            t = np.array([self._to_seconds(x) for x in t_raw], dtype=float)
            t = t - t[0]

        # rotations
        rx = self._col_or_zero(df, ["rot_x","rotation x","rotx","roll","rx"])
        ry = self._col_or_zero(df, ["rot_y","rotation y","roty","pitch","ry"])
        rz = self._col_or_zero(df, ["rot_z","rotation z","rotz","yaw","rz"])
        rot = np.vstack([rx, ry, rz]).T
        if not radians:
            rot = np.deg2rad(rot)

        # positions
        px = self._col_or_zero(df, ["pos_x","position x","x"])
        py = self._col_or_zero(df, ["pos_y","position y","y"])
        pz = self._col_or_zero(df, ["pos_z","position z","z"])
        pos = np.vstack([px, py, pz]).T
        return t, rot, pos

    @staticmethod
    def _find_col(df, names):
        low = {c.lower(): c for c in df.columns}
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        for c in df.columns:
            if any(n.lower() in c.lower() for n in names):
                return c
        return None

    @staticmethod
    def _col_or_zero(df, names):
        col = TimeSync._find_col(df, names)
        if col is None:
            return np.zeros(len(df))
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()

    # ---------- estimators ----------
    def _estimate_offset_by_nearest(self, t_ref, t_mov, search=(-10,10,0.005)):
        tmin, tmax, step = search
        taus = np.arange(tmin, tmax + 1e-12, step, dtype=float)
        best = (None, np.inf)
        t_ref = np.asarray(t_ref, dtype=float)
        t_mov = np.asarray(t_mov, dtype=float)
        for tau in taus:
            t_shift = t_mov + tau
            idx = np.searchsorted(t_shift, t_ref)
            idx0 = np.clip(idx, 0, len(t_shift)-1)
            idxm = np.clip(idx-1, 0, len(t_shift)-1)
            d0 = np.abs(t_shift[idx0] - t_ref)
            dm = np.abs(t_shift[idxm] - t_ref)
            d  = np.minimum(d0, dm)
            score = float(np.mean(d))
            if score < best[1]:
                best = (tau, score)
        return best  # (tau_to_add, mean_abs_error)

    def _estimate_offset_omega(self, t_mocap, rot_m, t_imu, w_imu, grid_dt=0.01, tau_search=10.0):
        """
        Estimate MoCapâ†”IMU time shift by comparing |Ï‰| magnitudes.
        Returns tau_to_add_to_IMU (i.e., add this to IMU to align with MoCap).
        Note: The caller typically flips sign to add the shift to MoCap instead.
        """
        # ---- Basic guards ----
        if w_imu is None or len(t_imu) < 2 or len(t_mocap) < 2:
            return 0.0

        # Ensure positive grid_dt
        if not np.isfinite(grid_dt) or grid_dt <= 0:
            grid_dt = 0.01

        # Sanitize & sort time-series (finite, strictly increasing)
        def sanitize_time_series(t, X=None):
            t = np.asarray(t, dtype=float)
            finite = np.isfinite(t)
            if X is not None:
                X = np.asarray(X)
                finite = finite & np.all(np.isfinite(X), axis=1)
            t = t[finite]
            if X is not None:
                X = X[finite]

            if t.size == 0:
                return np.array([]), None if X is None else np.empty((0, X.shape[1]))

            idx = np.argsort(t)
            t = t[idx]
            if X is not None:
                X = X[idx]

            if t.size > 1:
                keep = np.concatenate(([True], np.diff(t) > 0))
                t = t[keep]
                if X is not None:
                    X = X[keep]
            return t, X

        t_imu_s, w_imu_s = sanitize_time_series(t_imu, w_imu)
        t_moc_s, rot_m_s = sanitize_time_series(t_mocap, rot_m)

        if t_imu_s.size < 2 or t_moc_s.size < 2:
            return 0.0

        # ---- Overlap window ----
        t0 = max(t_imu_s[0], t_moc_s[0])
        t1 = min(t_imu_s[-1], t_moc_s[-1])

        # If no overlap or too short, bail out gracefully
        if not np.isfinite(t0) or not np.isfinite(t1) or (t1 <= t0):
            return 0.0
        if (t1 - t0) < 3.0 * grid_dt:
            return 0.0

        # Robust against float step issues
        n_pts = int((t1 - t0) / grid_dt) + 1
        if n_pts < 5:
            return 0.0
        grid = np.linspace(t0, t1, n_pts, dtype=float)

        # ---- Compute angular velocity from mocap rotations via time gradient ----
        try:
            omega_m = np.gradient(rot_m_s, t_moc_s, axis=0)  # (N,3)
        except Exception:
            omega_m = np.vstack([
                np.gradient(rot_m_s[:, 0], t_moc_s),
                np.gradient(rot_m_s[:, 1], t_moc_s),
                np.gradient(rot_m_s[:, 2], t_moc_s)
            ]).T

        # ---- Interpolate |Ï‰| magnitudes onto the common grid ----
        mag_m = np.linalg.norm(omega_m, axis=1)
        mag_i = np.linalg.norm(w_imu_s, axis=1)

        mag_m_g = np.interp(grid, t_moc_s, mag_m)
        mag_i_g = np.interp(grid, t_imu_s, mag_i)

        # Demean
        mag_m_g = mag_m_g - np.nanmean(mag_m_g)
        mag_i_g = mag_i_g - np.nanmean(mag_i_g)

        # ---- Cross-correlation over Â±tau_search ----
        max_lag = max(1, int(round(tau_search / grid_dt)))
        best = (0, -np.inf)

        def safe_norm(x):
            n = np.linalg.norm(x)
            return 1.0 if n == 0.0 or not np.isfinite(n) else n

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                a = mag_m_g[-lag:]
                b = mag_i_g[:len(a)]
            elif lag > 0:
                a = mag_m_g[:len(mag_m_g) - lag]
                b = mag_i_g[lag:lag + len(a)]
            else:
                a = mag_m_g
                b = mag_i_g

            if len(a) < 5:
                continue

            denom = safe_norm(a) * safe_norm(b)
            if denom == 0 or not np.isfinite(denom):
                continue

            c = float(np.dot(a, b) / denom)
            if c > best[1]:
                best = (lag, c)

        lag = best[0]
        tau = float(lag * grid_dt)  # interpretation: add tau to IMU â†’ align to MoCap
        return tau


def make_clahe(clip=3.0, grid=8):
    import cv2
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))


def apply_clahe(img_gray, clahe):
    return clahe.apply(img_gray)


def run_fast(vo):
    """Runs your existing FAST/LK pipeline exactly as you already do in main()."""
    if len(vo.gt_poses) > 0:
        T0 = vo.gt_poses[0]
        vo.initialize_isam_with_prior(T0_4x4=T0)
    else:
        vo.initialize_isam_with_prior(T0_4x4=None)

    gt_path, fused_path, errs = [], [], []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="poses")):
        if i == 0:
            est_pose = vo.isam.calculateEstimate().atPose3(X(0))
        else:
            T_km1_k = vo.get_pose(i)
            t_km1 = vo.cam_times[i-1]
            t_k   = vo.cam_times[i]
            est_pose = vo.fuse_vo_imu_step(T_km1_k, t_km1=t_km1, t_k=t_k)

        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        E = est_pose.matrix()
        E_plot = remap_pose(E)
        fused_path.append((E[0, 3], E[2, 3]))

        ex, ez = E[0, 3], E[2, 3]
        gx, gz = gt_pose[0, 3], gt_pose[2, 3]
        errs.append(float(np.hypot(ex - gx, ez - gz)))

    return gt_path, fused_path, errs


def run_sequence_with_backend(vo, backend, preprocess=None):
    # ---- 1) Clamp all streams to a common N ----
    nL  = len(vo.images_l)
    nR  = len(vo.images_r)
    nGT = len(vo.gt_poses) if getattr(vo, "gt_poses", None) else 10**9
    nT  = len(vo.cam_times) if getattr(vo, "cam_times", None) else 10**9
    N   = min(nL, nR, nGT, nT)

    images_l = vo.images_l[:N]
    images_r = vo.images_r[:N]
    cam_times = vo.cam_times[:N] if getattr(vo, "cam_times", None) else [None]*N
    gt_poses  = vo.gt_poses[:N]  if getattr(vo, "gt_poses", None) else [None]*N

    print(f"[frames] using N={N} (L={nL}, R={nR}, GT={nGT}, T={nT})")

    # ---- 2) ISAM prior (use GT[0] if present) ----
    if gt_poses[0] is not None:
        vo.initialize_isam_with_prior(T0_4x4=gt_poses[0])
    else:
        vo.initialize_isam_with_prior(T0_4x4=None)

    gt_path, fused_path, errs = [], [], []

    # ---- 3) First estimate from ISAM state ----
    est_pose = vo.isam.calculateEstimate().atPose3(X(0))
    E = est_pose.matrix()
    E_plot = remap_pose(E)
    if gt_poses[0] is not None:
        g0 = gt_poses[0]
        gt_path.append((g0[0,3], g0[2,3]))
        fused_path.append((E[0,3], E[2,3]))
        errs.append(float(np.hypot(E[0,3]-g0[0,3], E[2,3]-g0[2,3])))

    # ---- 4) Iterate frames 1..N-1 (use common N) ----
    from tqdm import tqdm
    for i in tqdm(range(1, N), unit="poses"):
        img_prev_l, img_prev_r = images_l[i-1], images_r[i-1]
        img_curr_l, img_curr_r = images_l[i],   images_r[i]

        if preprocess:
            img_prev_l = preprocess(img_prev_l)
            img_prev_r = preprocess(img_prev_r)
            img_curr_l = preprocess(img_curr_l)
            img_curr_r = preprocess(img_curr_r)

        # Detect + match on LEFT images (prev vs curr)
        pts1, des1 = backend.detect_and_describe(img_prev_l)
        pts2, des2 = backend.detect_and_describe(img_curr_l)
        idx1, idx2, _ = backend.match(des1, des2)

        if len(idx1) < 6:
            T_km1_k = np.eye(4)
        else:
            q1_l = pts1[idx1]   # prev-left keypoints
            q2_l = pts2[idx2]   # curr-left keypoints

            # Compute BOTH disparities explicitly for the two stereo pairs
            disp_prev = vo.disparity.compute(img_prev_l, img_prev_r).astype(np.float32) / 16.0
            disp_curr = vo.disparity.compute(img_curr_l, img_curr_r).astype(np.float32) / 16.0
            vo.disparities.append(disp_curr)  # optional history

            # Get corresponding right-image points using your helper
            q1_l_f, q1_r, q2_l_f, q2_r = vo.calculate_right_qs(q1_l, q2_l, disp_prev, disp_curr)

            if q1_l_f.shape[0] < 6:
                T_km1_k = np.eye(4)
            else:
                Q1, Q2 = vo.calc_3d(q1_l_f, q1_r, q2_l_f, q2_r)
                T_km1_k = vo.estimate_pose(q1_l_f, q2_l_f, Q1, Q2)

        t_km1 = cam_times[i-1]
        t_k   = cam_times[i]
        est_pose = vo.fuse_vo_imu_step(T_km1_k, t_km1=t_km1, t_k=t_k)
        E = est_pose.matrix()
        E_plot = remap_pose(E)
        g = gt_poses[i]
        if g is not None:
            gt_path.append((g[0,3], g[2,3]))
            fused_path.append((E[0,3], E[2,3]))
            errs.append(float(np.hypot(E[0,3]-g[0,3], E[2,3]-g[2,3])))

    return gt_path, fused_path, errs



def main():
    # -------- MoCap load --------
    helper    = MocapHelper()
    mocap_df  = helper._load_xy_from_motive_csv(Path(MOCAP_CSV), body_name="HUSKY")
    mocap_t_rel = mocap_df["t"].to_numpy(dtype=float)   # seconds, relative to capture start
    mocap_x   = mocap_df["x"].to_numpy(dtype=float)
    mocap_z   = -mocap_df["z"].to_numpy(dtype=float)

    # -------- Camera & IMU (Absolute times) --------
    # Camera times (absolute seconds)
    t_cam_abs = load_cam_times_from_zed_csv(LEFT_DATACSV, images_dir=LEFT_DIR)
    
    # IMU (absolute times)
    imu_samples = load_imu_csv_ns(IMU_CSV, has_header=True)
    t_imu_abs = np.array([s.timestamp for s in imu_samples], dtype=float)

    # ========== TIME ALIGNMENT ==========
    print("=== TIME ALIGNMENT ===")
    print(f"Camera absolute: {len(t_cam_abs)} frames, {t_cam_abs[0]:.6f} to {t_cam_abs[-1]:.6f}")
    print(f"IMU absolute:    {len(t_imu_abs)} samples, {t_imu_abs[0]:.6f} to {t_imu_abs[-1]:.6f}")
    print(f"MoCap relative:  {len(mocap_t_rel)} poses, {mocap_t_rel[0]:.3f} to {mocap_t_rel[-1]:.3f}")

    # Strategy: Convert everything to be relative to first camera frame
    # This assumes camera and IMU are already synchronized, MoCap needs time offset
    
    # Find time offset between MoCap and camera using motion correlation
    TIME_OFFSET = find_time_offset_between_mocap_camera(mocap_t_rel, mocap_x, mocap_z, 
                                                       t_cam_abs, imu_samples)
    
    print(f"Using time offset: {TIME_OFFSET:.3f} seconds (MoCap starts {TIME_OFFSET:.3f}s after camera)")

    # Convert MoCap to camera-relative time
    mocap_t_aligned = mocap_t_rel + TIME_OFFSET

    # Convert all to relative time (relative to first camera frame)
    t_cam_rel = t_cam_abs - t_cam_abs[0]  # Camera relative to first frame
    t_imu_rel = t_imu_abs - t_cam_abs[0]  # IMU relative to first camera frame
    mocap_t_final = mocap_t_aligned       # MoCap already relative after alignment
    
    # Update IMU samples with relative timestamps
    for i, sample in enumerate(imu_samples):
        sample.timestamp = t_imu_rel[i]

    print(f"\nAfter time alignment (relative to first camera frame):")
    print(f"Camera: {t_cam_rel[0]:.3f} to {t_cam_rel[-1]:.3f} s")
    print(f"IMU:    {t_imu_rel[0]:.3f} to {t_imu_rel[-1]:.3f} s")
    print(f"MoCap:  {mocap_t_final[0]:.3f} to {mocap_t_final[-1]:.3f} s")

    # -------- Align GT to Camera Frames --------
    gt_poses_from_mocap, valid_camera_indices = align_gt_to_camera_by_time(
        mocap_t_final, mocap_x, mocap_z, t_cam_rel, max_time_diff=0.05
    )

    print(f"Aligned {len(gt_poses_from_mocap)} GT poses to {len(t_cam_rel)} camera frames")

    # -------- VO construction --------
    vo = VisualOdometry(
        calib_path=CALIB_PATH,
        left_dir=LEFT_DIR,
        right_dir=RIGHT_DIR,
        poses_path=POSES_PATH,
        gt_poses=gt_poses_from_mocap
    )

    # Use only the frames that have GT alignment
    vo.images_l = [vo.images_l[i] for i in valid_camera_indices]
    vo.images_r = [vo.images_r[i] for i in valid_camera_indices]
    vo.cam_times = [t_cam_rel[i] for i in valid_camera_indices]
    
    # Also attach IMU with relative timestamps
    vo.attach_imu(imu_samples=imu_samples, cam_times=list(vo.cam_times))

    print(f"Final aligned streams: {len(vo.images_l)} frames, {len(vo.cam_times)} timestamps, {len(vo.gt_poses)} GT poses")

    # Initial bias + attitude from IMU
    bias0, R0_np = _estimate_bias_and_R0_from_imu(imu_samples, t_window=1.0)
    vo.bias0 = bias0
    vo.accum = PreintegratedImuMeasurements(vo.params, vo.bias0)

    # Initial pose prior (gravity-aligned attitude, zero position)
    T0 = np.eye(4, dtype=float); T0[:3, :3] = R0_np
    vo.initialize_isam_with_prior(T0_4x4=T0)

    # Camera->IMU extrinsic
    R_cam_imu = np.eye(3)
    t_cam_imu = np.array([0.06, 0.0, 0.0], float)
    vo.set_cam_imu_extrinsics(R_cam_imu, t_cam_imu)

    # ========== DEBUGGING SECTION ==========
    print("\n" + "="*50)
    print("STARTING COMPREHENSIVE DEBUGGING")
    print("="*50)

    def debug_feature_matching(vo, backend, frame_idx=0):
        """Debug feature matching between consecutive frames"""
        img1 = vo.images_l[frame_idx]
        img2 = vo.images_l[frame_idx + 1]
        
        # Detect and describe features
        pts1, des1 = backend.detect_and_describe(img1)
        pts2, des2 = backend.detect_and_describe(img2)
        
        print(f"Frame {frame_idx}: Found {len(pts1)} keypoints in img1, {len(pts2)} in img2")
        
        # Match features
        idx1, idx2, good_matches = backend.match(des1, des2)
        
        print(f"Found {len(good_matches)} good matches")
        
        return len(good_matches), pts1, pts2, idx1, idx2

    def debug_imu_preintegration(vo, frame_idx=1):
        """Debug IMU preintegration between frames"""
        if len(vo.imu) == 0:
            print("No IMU data available")
            return
        
        t_km1 = vo.cam_times[frame_idx - 1]
        t_k = vo.cam_times[frame_idx]
        
        print(f"Camera times: t_{frame_idx-1}={t_km1:.3f}s, t_{frame_idx}={t_k:.3f}s")
        print(f"Time delta: {t_k - t_km1:.3f}s")
        
        # Find IMU samples in this interval
        imu_in_interval = [s for s in vo.imu if t_km1 < s.timestamp <= t_k]
        print(f"Found {len(imu_in_interval)} IMU samples in interval")
        
        if len(imu_in_interval) > 0:
            # Print first few samples
            for i, sample in enumerate(imu_in_interval[:3]):
                print(f"  IMU sample {i}: t={sample.timestamp:.3f}s, "
                      f"accel={sample.accel}, gyro={sample.gyro}")
        
        # Test preintegration
        try:
            # Create a fresh preintegrator
            test_accum = PreintegratedImuMeasurements(vo.params, vo.bias0)
            
            # Preintegrate
            vo.preintegrate_between(test_accum, t_km1, t_k, 
                                  imu_samples=vo.imu, 
                                  imu_times=vo.imu_times,
                                  R_align=R_ALIGN_IMU_TO_VO)
            
            print("Preintegration successful!")
            print(f"Delta pose: {test_accum.deltaPose().matrix()}")
            print(f"Delta velocity: {test_accum.deltaVelocity()}")
            print(f"Delta position: {test_accum.deltaPosition()}")
            
        except Exception as e:
            print(f"Preintegration failed: {e}")

    def debug_isam_graph_simple(vo, iteration=0):
        """Debug ISAM2 state and factor graph - CORRECTED VERSION"""
        print(f"\n=== ISAM2 Debug - Iteration {iteration} ===")
        
        print(f"ISAM initialized: {vo.has_initialized_isam}")
        print(f"Current key (k): {vo.k}")
        
        if vo.has_initialized_isam:
            try:
                result = vo.isam.calculateEstimate()
                
                print("Current estimates (simplified):")
                print(f"Number of values: {result.size()}")
                
                # List all keys and their types
                keys = list(result.keys())
                print(f"Keys present: {keys}")
                
                # Try to read specific expected keys
                for i in range(vo.k + 1):  # Check poses up to current k
                    pose_key = X(i)
                    vel_key = V(i)
                    bias_key = B(i // 5 if i > 0 else 0)  # Your bias key pattern
                    
                    if pose_key in keys:
                        try:
                            pose = result.atPose3(pose_key)
                            pos = pose.translation()
                            print(f"  X({i}): pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                        except Exception as e:
                            print(f"  X({i}): Error reading pose - {e}")
                    
                    if vel_key in keys:
                        try:
                            vel = result.atVector(vel_key)
                            print(f"  V({i}): vel=({vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f})")
                        except:
                            pass
                    
                    if bias_key in keys:
                        try:
                            bias = result.atConstantBias(bias_key)
                            print(f"  B({i//5}): accel_bias=({bias.accelerometer()[0]:.3f}, {bias.accelerometer()[1]:.3f}, {bias.accelerometer()[2]:.3f})")
                        except:
                            pass
                            
            except Exception as e:
                print(f"Error getting estimates: {e}")
        
        print(f"Graph size: {vo.graph.size()}")
        print(f"Initial values size: {vo.initial.size()}")

    def check_imu_units(vo):
        """Check if IMU units are reasonable"""
        if len(vo.imu) == 0:
            return
        
        # Check accelerometer data (should be around 9.81 m/s² when stationary)
        accel_norms = [np.linalg.norm(s.accel) for s in vo.imu[:100]]
        avg_accel_norm = np.mean(accel_norms)
        print(f"Average accelerometer norm: {avg_accel_norm:.3f}")
        print(f"Expected gravity: ~9.81 m/s²")
        
        # Check gyroscope data (should be in rad/s typically)
        gyro_norms = [np.linalg.norm(s.gyro) for s in vo.imu[:100]]
        avg_gyro_norm = np.mean(gyro_norms)
        print(f"Average gyroscope norm: {avg_gyro_norm:.3f}")
        print("If this is > 10, gyro might be in deg/s instead of rad/s")

    # Run initial debugging
    print("\n1. Checking IMU units...")
    check_imu_units(vo)

    print("\n2. Checking ISAM2 initialization...")
    debug_isam_graph_simple(vo, iteration=0)

    # ---------- ORB run ----------
    if len(vo.gt_poses) > 0:
        # Make a second VO for ORB front-end with the SAME timings/IMU
        vo_orb = VisualOdometry(
            calib_path=CALIB_PATH,
            left_dir=LEFT_DIR,
            right_dir=RIGHT_DIR,
            poses_path=POSES_PATH,
            gt_poses=vo.gt_poses  # Use the same aligned GT poses
        )
        vo_orb.images_l = vo.images_l.copy()
        vo_orb.images_r = vo.images_r.copy()
        vo_orb.gt_poses = vo.gt_poses
        vo_orb.cam_times = vo.cam_times
        vo_orb.attach_imu(imu_samples=imu_samples, cam_times=list(vo_orb.cam_times))

        # Same init for ORB path
        vo_orb.bias0 = bias0
        vo_orb.accum = PreintegratedImuMeasurements(vo_orb.params, vo_orb.bias0)
        vo_orb.set_cam_imu_extrinsics(R_cam_imu, t_cam_imu)
        vo_orb.initialize_isam_with_prior(T0_4x4=T0)

        orb = ORBBackend(ratio=0.75)
        clahe = make_clahe(clip=3.0, grid=8)

        print("\n3. Testing feature matching...")
        num_matches, pts1, pts2, idx1, idx2 = debug_feature_matching(vo_orb, orb, frame_idx=0)

        print("\n4. Testing IMU preintegration...")
        debug_imu_preintegration(vo_orb, frame_idx=1)

        print("\n5. Running first few iterations with detailed debugging...")
        # Run a few iterations with detailed debugging
        gt_orb, traj_orb, err_orb = [], [], []
        
        # Add initial pose
        est_pose = vo_orb.isam.calculateEstimate().atPose3(X(0))
        E = est_pose.matrix()
        if vo_orb.gt_poses[0] is not None:
            g0 = vo_orb.gt_poses[0]
            gt_orb.append((g0[0,3], g0[2,3]))
            traj_orb.append((E[0,3], E[2,3]))
            err_orb.append(float(np.hypot(E[0,3]-g0[0,3], E[2,3]-g0[2,3])))

        from tqdm import tqdm
        n_frames_debug = min(10, len(vo_orb.images_l))
        for i in tqdm(range(1, n_frames_debug), unit="poses"):
            print(f"\n--- Processing frame {i} ---")
            
            img_prev_l, img_prev_r = vo_orb.images_l[i-1], vo_orb.images_r[i-1]
            img_curr_l, img_curr_r = vo_orb.images_l[i], vo_orb.images_r[i]

            if clahe:
                img_prev_l = apply_clahe(img_prev_l, clahe)
                img_prev_r = apply_clahe(img_prev_r, clahe)
                img_curr_l = apply_clahe(img_curr_l, clahe)
                img_curr_r = apply_clahe(img_curr_r, clahe)

            # Detect + match on LEFT images (prev vs curr)
            pts1, des1 = orb.detect_and_describe(img_prev_l)
            pts2, des2 = orb.detect_and_describe(img_curr_l)
            idx1, idx2, good_matches = orb.match(des1, des2)

            print(f"Frame {i}: {len(good_matches)} matches found")

            if len(idx1) < 6:
                T_km1_k = np.eye(4)
                print("WARNING: Not enough matches, using identity transform")
            else:
                q1_l = pts1[idx1]   # prev-left keypoints
                q2_l = pts2[idx2]   # curr-left keypoints

                # Compute BOTH disparities explicitly for the two stereo pairs
                disp_prev = vo_orb.disparity.compute(vo_orb.images_l[i-1], vo_orb.images_r[i-1]).astype(np.float32) / 16.0
                disp_curr = vo_orb.disparity.compute(vo_orb.images_l[i], vo_orb.images_r[i]).astype(np.float32) / 16.0
                vo_orb.disparities.append(disp_curr)  # optional history

                # Get corresponding right-image points using your helper
                q1_l_f, q1_r, q2_l_f, q2_r = vo_orb.calculate_right_qs(q1_l, q2_l, disp_prev, disp_curr)

                print(f"After stereo matching: {len(q1_l_f)} 3D points")

                if q1_l_f.shape[0] < 6:
                    T_km1_k = np.eye(4)
                    print("WARNING: Not enough 3D points after stereo matching, using identity transform")
                else:
                    Q1, Q2 = vo_orb.calc_3d(q1_l_f, q1_r, q2_l_f, q2_r)
                    T_km1_k = vo_orb.estimate_pose(q1_l_f, q2_l_f, Q1, Q2)
                    print(f"Estimated transformation:\n{T_km1_k}")

            t_km1 = vo_orb.cam_times[i-1]
            t_k   = vo_orb.cam_times[i]
            
            print(f"Fusing with IMU (t_km1={t_km1:.3f}, t_k={t_k:.3f})")
            est_pose = vo_orb.fuse_vo_imu_step(T_km1_k, t_km1=t_km1, t_k=t_k)
            
            E = est_pose.matrix()
            g = vo_orb.gt_poses[i]
            if g is not None:
                gt_orb.append((g[0,3], g[2,3]))
                traj_orb.append((E[0,3], E[2,3]))
                err_orb.append(float(np.hypot(E[0,3]-g[0,3], E[2,3]-g[2,3])))
                print(f"Fused pose - X: {E[0,3]:.3f}, Z: {E[2,3]:.3f}")
                print(f"GT pose - X: {g[0,3]:.3f}, Z: {g[2,3]:.3f}")
                print(f"Error: {err_orb[-1]:.3f}")

            # Debug ISAM state after update
            debug_isam_graph_simple(vo_orb, iteration=i)

        # Continue with remaining frames if debugging was successful
        if len(vo_orb.images_l) > n_frames_debug:
            print(f"\n6. Continuing with remaining {len(vo_orb.images_l) - n_frames_debug} frames...")
            for i in tqdm(range(n_frames_debug, len(vo_orb.images_l)), unit="poses"):
                # Regular processing without detailed debug output
                img_prev_l, img_prev_r = vo_orb.images_l[i-1], vo_orb.images_r[i-1]
                img_curr_l, img_curr_r = vo_orb.images_l[i], vo_orb.images_r[i]

                if clahe:
                    img_prev_l = apply_clahe(img_prev_l, clahe)
                    img_prev_r = apply_clahe(img_prev_r, clahe)
                    img_curr_l = apply_clahe(img_curr_l, clahe)
                    img_curr_r = apply_clahe(img_curr_r, clahe)

                pts1, des1 = orb.detect_and_describe(img_prev_l)
                pts2, des2 = orb.detect_and_describe(img_curr_l)
                idx1, idx2, _ = orb.match(des1, des2)

                if len(idx1) < 6:
                    T_km1_k = np.eye(4)
                else:
                    q1_l = pts1[idx1]
                    q2_l = pts2[idx2]

                    disp_prev = vo_orb.disparity.compute(vo_orb.images_l[i-1], vo_orb.images_r[i-1]).astype(np.float32) / 16.0
                    disp_curr = vo_orb.disparity.compute(vo_orb.images_l[i], vo_orb.images_r[i]).astype(np.float32) / 16.0
                    vo_orb.disparities.append(disp_curr)

                    q1_l_f, q1_r, q2_l_f, q2_r = vo_orb.calculate_right_qs(q1_l, q2_l, disp_prev, disp_curr)

                    if q1_l_f.shape[0] < 6:
                        T_km1_k = np.eye(4)
                    else:
                        Q1, Q2 = vo_orb.calc_3d(q1_l_f, q1_r, q2_l_f, q2_r)
                        T_km1_k = vo_orb.estimate_pose(q1_l_f, q2_l_f, Q1, Q2)

                t_km1 = vo_orb.cam_times[i-1]
                t_k   = vo_orb.cam_times[i]
                est_pose = vo_orb.fuse_vo_imu_step(T_km1_k, t_km1=t_km1, t_k=t_k)
                
                E = est_pose.matrix()
                g = vo_orb.gt_poses[i]
                if g is not None:
                    gt_orb.append((g[0,3], g[2,3]))
                    traj_orb.append((E[0,3], E[2,3]))
                    err_orb.append(float(np.hypot(E[0,3]-g[0,3], E[2,3]-g[2,3])))

        print("First 10 poses (x,z) ORB:", traj_orb[:10])
        print("First 10 GT (x,z):",        gt_orb[:10])
        print(f"[ORB ] mean |pos| error (x–z): {float(np.mean(err_orb)) if len(err_orb) else float('nan'):.4f}")

        plotting.visualize_paths(
            gt_orb, traj_orb, "Stereo VO (ORB front-end)",
            file_out=os.path.basename(LEFT_DIR.rstrip(os.sep)) + "_orb_vo.html"
        )
    else:
        print("No GT poses provided; VO runs skipped.")


# ========== TIME ALIGNMENT FUNCTIONS ==========

def find_time_offset_between_mocap_camera(mocap_t_rel, mocap_x, mocap_z, t_cam_abs, imu_samples, search_range=(-2.0, 2.0), step=0.1):
    """
    Find time offset between MoCap and camera by comparing motion patterns.
    Returns the offset to ADD to MoCap time to align with camera.
    """
    print("Finding time offset between MoCap and camera...")
    
    # Convert camera and IMU to temporary relative time for correlation
    t_cam_temp = t_cam_abs - t_cam_abs[0]
    t_imu_temp = np.array([s.timestamp for s in imu_samples]) - t_cam_abs[0]
    
    # Compute motion signals
    # MoCap: velocity magnitude from position
    mocap_pos_mag = np.sqrt(mocap_x**2 + mocap_z**2)
    mocap_vel = np.gradient(mocap_pos_mag, mocap_t_rel)
    
    # IMU: gyro magnitude as proxy for motion
    imu_gyro_mag = np.array([np.linalg.norm(s.gyro) for s in imu_samples])
    
    # Resample to common time grid
    common_dt = 0.02  # 50 Hz
    common_t_max = min(mocap_t_rel[-1], t_imu_temp[-1], t_cam_temp[-1])
    common_t = np.arange(0, common_t_max, common_dt)
    
    if len(common_t) < 20:
        print("Warning: Not enough common time for correlation, using offset=0")
        return 0.0
    
    # Interpolate signals to common grid
    mocap_vel_interp = np.interp(common_t, mocap_t_rel, mocap_vel, left=0, right=0)
    imu_gyro_interp = np.interp(common_t, t_imu_temp, imu_gyro_mag, left=0, right=0)
    
    # Normalize signals
    mocap_vel_interp = (mocap_vel_interp - np.mean(mocap_vel_interp)) / (np.std(mocap_vel_interp) + 1e-9)
    imu_gyro_interp = (imu_gyro_interp - np.mean(imu_gyro_interp)) / (np.std(imu_gyro_interp) + 1e-9)
    
    # Cross-correlation to find best offset
    best_offset = 0.0
    best_correlation = -np.inf
    
    for offset in np.arange(search_range[0], search_range[1] + step, step):
        # Shift MoCap signal
        shifted_mocap = np.interp(common_t - offset, common_t, mocap_vel_interp, left=0, right=0)
        
        # Compute correlation on overlapping part
        valid = np.isfinite(shifted_mocap) & np.isfinite(imu_gyro_interp)
        if np.sum(valid) > 20:  # Need enough valid points
            correlation = np.corrcoef(shifted_mocap[valid], imu_gyro_interp[valid])[0, 1]
            if np.isfinite(correlation) and correlation > best_correlation:
                best_correlation = correlation
                best_offset = offset
    
    print(f"Best time offset: {best_offset:.3f} s (correlation: {best_correlation:.3f})")
    return best_offset


def align_gt_to_camera_by_time(mocap_t, mocap_x, mocap_z, camera_times, max_time_diff=0.05):
    """
    Align GT poses to camera frames using time matching.
    
    Args:
        mocap_t: MoCap times (relative to same reference as camera_times)
        mocap_x, mocap_z: MoCap positions  
        camera_times: Camera frame times
        max_time_diff: Maximum allowed time difference for matching
    
    Returns:
        aligned_poses: GT poses for each camera frame
        valid_indices: Camera frame indices that have valid GT
    """
    aligned_poses = []
    valid_indices = []
    time_errors = []
    
    for i, cam_time in enumerate(camera_times):
        # Find nearest MoCap pose in time
        time_diffs = np.abs(mocap_t - cam_time)
        nearest_idx = np.argmin(time_diffs)
        min_time_diff = time_diffs[nearest_idx]
        
        if min_time_diff < max_time_diff:
            # Valid match found
            x, z = mocap_x[nearest_idx], mocap_z[nearest_idx]
            tvec = np.array([x, 0.0, z], float)
            T = np.eye(4, dtype=float)
            T[:3, 3] = tvec
            aligned_poses.append(T)
            valid_indices.append(i)
            time_errors.append(min_time_diff)
        # else: Skip this frame if no good match
    
    if time_errors:
        print(f"Time alignment: {len(aligned_poses)}/{len(camera_times)} camera frames matched")
        print(f"Time errors - mean: {np.mean(time_errors):.4f}s, max: {np.max(time_errors):.4f}s")
    else:
        print("WARNING: No frames could be time-aligned!")
    
    return aligned_poses, valid_indices

def save_traj_csv(csv_path, est_xz, gt_xz=None, errs=None, times=None, extra_cols=None):
    """
    Save trajectories to CSV.

    Parameters
    ----------
    csv_path : str or Path
        Output file path.
    est_xz : list[tuple(float,float)] or np.ndarray (N,2)
        Estimated positions (x,z) per pose.
    gt_xz : list[tuple(float,float)] or np.ndarray (N,2), optional
        Ground-truth positions (x,z) per pose. If None, columns left blank.
    errs : list[float] or np.ndarray (N,), optional
        Per-pose x–z errors. If None and gt_xz is provided, it's computed.
    times : list[float] or np.ndarray (N,), optional
        Timestamp per pose. If provided, saved in the CSV.
    extra_cols : dict[str, list or np.ndarray], optional
        Any extra per-pose columns to include (e.g., {'y_est': y_est, 'y_gt': y_gt}).

    Notes
    -----
    - All arrays/lists are truncated to the common minimum length.
    - If errs is None and gt_xz is provided, errs is computed as sqrt((dx)^2 + (dz)^2).
    """
    csv_path = Path(csv_path)
    est_xz = list(est_xz)
    N = len(est_xz)

    # Normalize optionals
    gt_xz  = list(gt_xz)  if gt_xz  is not None else [None]*N
    errs   = list(errs)   if errs   is not None else [None]*N
    times  = list(times)  if times  is not None else [None]*N
    extra_cols = extra_cols or {}

    # Determine common length
    lens = [N, len(gt_xz), len(errs), len(times)] + [len(v) for v in extra_cols.values()]
    Nmin = min(lens)

    # Truncate everything to Nmin
    est_xz  = est_xz[:Nmin]
    gt_xz   = gt_xz[:Nmin]
    errs    = errs[:Nmin]
    times   = times[:Nmin]
    extra_cols = {k: list(v)[:Nmin] for k, v in extra_cols.items()}

    # Compute errs if needed and GT present
    if any(e is None for e in errs) and all(g is not None for g in gt_xz):
        import math
        errs = [
            math.hypot(ex - gx, ez - gz)
            if (g is not None) else None
            for (ex, ez), (gx, gz), g in zip(est_xz, gt_xz, gt_xz)
        ]

    # Header
    fieldnames = ["idx", "time", "est_x", "est_z", "gt_x", "gt_z", "err_xz"]
    # Add any extra columns in stable order
    for k in extra_cols.keys():
        if k not in fieldnames:
            fieldnames.append(k)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(Nmin):
            ex, ez = est_xz[i]
            if gt_xz[i] is not None:
                gx, gz = gt_xz[i]
            else:
                gx = gz = ""
            row = {
                "idx": i,
                "time": times[i] if times[i] is not None else "",
                "est_x": float(ex),
                "est_z": float(ez),
                "gt_x":  "" if gx == "" else float(gx),
                "gt_z":  "" if gz == "" else float(gz),
                "err_xz": "" if errs[i] is None else float(errs[i]),
            }
            for k, v in extra_cols.items():
                row[k] = v[i]
            w.writerow(row)

    print(f"[save_traj_csv] wrote {Nmin} rows → {csv_path}")
if __name__ == "__main__":
    main()