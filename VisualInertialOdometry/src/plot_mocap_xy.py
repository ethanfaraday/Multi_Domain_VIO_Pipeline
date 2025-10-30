# Use like this from your current working directory:
# data_dir = 'KITTI_sequence_1'  # e.g., your sequence folder next to your script
# The code below has a main that calls:
# gt_runs = helper._load_mocap_csvs(data_dir + '/MOCAP') and then plots them.

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt


class MocapHelper:
    # ===================== PUBLIC API (matches your style) =====================
    def _load_mocap_csvs(self, mocap_dir: str, body_name: str = "HUSKY") -> List[pd.DataFrame]:
        """
        Load all Motive/OptiTrack CSVs in mocap_dir and return a list of DataFrames with columns ['t','x','z','rx','ry','rz'].
        Example: gt_runs = self._load_mocap_csvs(data_dir + '/MOCAP')
        """
        mocap_path = Path(mocap_dir)  # relative to current working directory
        if not mocap_path.exists():
            raise FileNotFoundError(f"MOCAP directory not found: {mocap_path}")

        csvs = sorted(mocap_path.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No .csv files found in {mocap_path}")

        out: List[pd.DataFrame] = []
        for csv_path in csvs:
            try:
                df_xy = self._load_xy_from_motive_csv(csv_path, body_name=body_name)
                df_xy.attrs["source"] = csv_path.name
                out.append(df_xy)
            except Exception as e:
                print(f"[WARN] Skipping {csv_path.name}: {e}")
        return out

    def plot_xy_list(self, xy_list: List[pd.DataFrame], title: str = "MOCAP XY Trajectories") -> None:
        """Plot Y vs X for each trajectory in xy_list."""
        if not xy_list:
            print("[WARN] No trajectories to plot.")
            return
        plt.figure()
        for df in xy_list:
            label = df.attrs.get("source", "run")
            plt.plot(df["x"].values, df["z"].values, linewidth=1.5, label=label)
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(title)
        plt.axis("equal")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    # ===================== INTERNALS: Motive parsing =====================
    def _detect_header_bottom_index(self, csv_path: Path) -> Optional[int]:
        """
        Return 0-based line index of the bottom header row that starts with 'Frame,Time (Seconds),...'.
        Motive CSVs have a 5-row header block ending with that line.
        """
        with csv_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                s = line.strip().lower().replace(" ", "")
                if s.startswith("frame,time(seconds)"):
                    return i
        return None

    def _read_motive_multiindex(self, csv_path: Path) -> pd.DataFrame:
        """
        Read a Motive CSV using its 5-row MultiIndex header:
            H-4: ,Type, Rigid Body, ...
            H-3: ,Name, <BodyName>, ...
            H-2: ,ID, <GUID>, ...
            H-1: , , Rotation/Position, ...
            H:   Frame, Time (Seconds), X, Y, Z, ...
        """
        h_bottom = self._detect_header_bottom_index(csv_path)
        if h_bottom is None or h_bottom < 4:
            raise RuntimeError(f"Could not detect Motive header structure in '{csv_path.name}'.")
        header_rows = [h_bottom - 4, h_bottom - 3, h_bottom - 2, h_bottom - 1, h_bottom]
        return pd.read_csv(csv_path, header=header_rows)

    def _extract_pose_data(self, df_mi: pd.DataFrame, body_name: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Extract time, rotation (rx, ry, rz), and position (x, y, z) from multi-index columns.
        """
        mi = df_mi.columns

        # Time column
        time_idx = None
        for i, col in enumerate(mi):
            if any(str(level).strip().lower() == "time (seconds)" for level in col):
                time_idx = i
                break
        if time_idx is None:
            raise RuntimeError("Could not find 'Time (Seconds)' column.")

        # Position X/Y/Z columns for the specified rigid body
        pos_x_idx = pos_y_idx = pos_z_idx = None
        rot_x_idx = rot_y_idx = rot_z_idx = None
        
        for i, col in enumerate(mi):
            levels = [str(x).strip() for x in col]
            if body_name in levels and "Position" in levels:
                if "X" in levels:
                    pos_x_idx = i
                elif "Y" in levels:
                    pos_y_idx = i
                elif "Z" in levels:
                    pos_z_idx = i
            elif body_name in levels and "Rotation" in levels:
                if "X" in levels:
                    rot_x_idx = i
                elif "Y" in levels:
                    rot_y_idx = i
                elif "Z" in levels:
                    rot_z_idx = i

        if pos_x_idx is None or pos_z_idx is None:
            raise RuntimeError(f"Could not identify Position X/Z columns for body '{body_name}'.")
        if rot_x_idx is None or rot_z_idx is None:
            print(f"Warning: Could not identify Rotation X/Z columns for body '{body_name}'. Using zero rotation.")
            # Set default zero rotations if not found
            rot_x_idx = rot_y_idx = rot_z_idx = time_idx  # Use time column as placeholder, will be set to zero

        t = pd.to_numeric(df_mi.iloc[:, time_idx], errors="coerce")
        x = pd.to_numeric(df_mi.iloc[:, pos_x_idx], errors="coerce")
        z = pd.to_numeric(df_mi.iloc[:, pos_z_idx], errors="coerce")
        
        # Extract y position if available, otherwise set to zero
        if pos_y_idx is not None:
            y = pd.to_numeric(df_mi.iloc[:, pos_y_idx], errors="coerce")
        else:
            y = pd.Series([0.0] * len(t))
            
        # Extract rotations
        if rot_x_idx is not None and rot_x_idx != time_idx:
            rx = pd.to_numeric(df_mi.iloc[:, rot_x_idx], errors="coerce")
            ry = pd.to_numeric(df_mi.iloc[:, rot_y_idx], errors="coerce") if rot_y_idx is not None else pd.Series([0.0] * len(t))
            rz = pd.to_numeric(df_mi.iloc[:, rot_z_idx], errors="coerce") if rot_z_idx is not None else pd.Series([0.0] * len(t))
        else:
            # Use zero rotations if not available
            rx = pd.Series([0.0] * len(t))
            ry = pd.Series([0.0] * len(t))
            rz = pd.Series([0.0] * len(t))

        return t, rx, ry, rz, x, y, z

    def _load_xy_from_motive_csv(self, csv_path: Path, body_name: str = "HUSKY") -> pd.DataFrame:
        """Load a single Motive CSV and return DataFrame with ['t','rx','ry','rz','x','y','z']."""
        df_mi = self._read_motive_multiindex(csv_path)
        t, rx, ry, rz, x, y, z = self._extract_pose_data(df_mi, body_name)
        return pd.DataFrame({"t": t, "rx": rx, "ry": ry, "rz": rz, "x": x, "y": y, "z": z}).dropna()


# ===================== MAIN (no CLI args; uses current directory) =====================
if __name__ == "__main__":
    # Set this to your sequence folder relative to *current working directory*
    data_dir = 'C:\VIOCODE\ComputerVision\VisualOdometry'          # <-- change this if needed
    body_name = 'HUSKY'                    # <-- change if your rigid body has a different name in Motive

    helper = MocapHelper()
    gt_runs = helper._load_mocap_csvs(data_dir + '/MOCAP', body_name=body_name)
    helper.plot_xy_list(gt_runs, title=f"{data_dir} MOCAP XY")