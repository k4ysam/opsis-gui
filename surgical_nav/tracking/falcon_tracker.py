"""FalconTracker: multi-camera pose tracking using the real_multicam.py pipeline.

Uses the exact same algorithm as real_multicam.py:
- DICT_APRILTAG_16h5 aruco dict
- Board points from .txt calibration files (MARKER_MAPPER=True, OFFSET=False)
- AprilTag corner refinement detector params
- Camera calibration loaded from JSON (same file as real_multicam.py)
- Per-camera pose_estimation.estimate_pose_board(), then anglesMean() fusion
- Same 4x4 matrix construction as the IGT block in real_multicam.py

Matches MockIGTLClient's signal interface so it can be swapped in transparently.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Union

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

_FALCON_TRACKING = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'falcon', 'Tracking')
)

_DEFAULT_CALIB_CAMERA = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..', '..',
        'falcon', 'Calibration', 'calib_files', 'camera',
        'T33_minpos_int_percam_cam1fixed.json',
    )
)
_DEFAULT_CALIB_TARGET = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..', '..',
        'falcon', 'Calibration', 'calib_files', 'tool',
        'T33_minpos_int_percam_cam1fixed_target.txt',
    )
)
_DEFAULT_CALIB_REF = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..', '..',
        'falcon', 'Calibration', 'calib_files', 'tool',
        'T33_minpos_int_percam_cam1fixed_ref.txt',
    )
)


def _ensure_falcon_path():
    if _FALCON_TRACKING not in sys.path:
        sys.path.insert(0, _FALCON_TRACKING)


# ---------------------------------------------------------------------------
# anglesMean — copied verbatim from real_multicam.py
# ---------------------------------------------------------------------------

def _two_angle_mean(theta1: float, theta2: float) -> float:
    if abs(theta1 - theta2) > 180:
        return ((theta1 + theta2) / 2 + 360) % 360 - 180
    return (theta1 + theta2) / 2


def _angles_mean(thetas: np.ndarray) -> float:
    if np.max(thetas) - np.min(thetas) > 180:
        avg = float(thetas[0])
        n = 1
        for theta in thetas[1:]:
            avg = _two_angle_mean(2 * avg * n / (n + 1), 2 * theta / (n + 1))
            n += 1
        return avg
    return float(np.mean(thetas))


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _FalconWorker(QObject):
    transform_ready = Signal(str, object)   # (name, 4×4 ndarray)
    status_changed  = Signal(str, str)      # (name, status)

    def __init__(
        self,
        sources: List[Union[int, str]],
        cam_ids: List[int],
        calib_camera: str,
        calib_target: str,
        calib_ref: str,
        fps: int,
    ):
        super().__init__()
        self._sources      = sources
        self._cam_ids      = cam_ids
        self._calib_camera = calib_camera
        self._calib_target = calib_target
        self._calib_ref    = calib_ref
        self._fps          = fps
        self._running      = False

    def run(self) -> None:
        self._running = True

        # --- Lazy imports ---
        try:
            import cv2
        except ImportError:
            print("[FalconTracker] OpenCV not installed — tracking disabled")
            return

        try:
            from scipy.spatial.transform import Rotation as R_scipy
        except ImportError:
            print("[FalconTracker] scipy not installed — tracking disabled")
            return

        _ensure_falcon_path()
        try:
            from util.pose_estimation import pose_estimation as PoseEstimator
        except ImportError as exc:
            print(f"[FalconTracker] FALCON utils not found: {exc}")
            return

        # --- ArUco dict — same as real_multicam.py ---
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

        # --- Detector params — same as real_multicam.py ---
        det_params = cv2.aruco.DetectorParameters()
        det_params.cornerRefinementMethod         = cv2.aruco.CORNER_REFINE_APRILTAG
        det_params.cornerRefinementMaxIterations  = 1000
        det_params.cornerRefinementMinAccuracy    = 0.001
        det_params.adaptiveThreshWinSizeStep      = 2
        det_params.adaptiveThreshWinSizeMax       = 30
        det_params.adaptiveThreshConstant         = 8
        detector = cv2.aruco.ArucoDetector(aruco_dict, det_params)

        # --- LM criteria — same as real_multicam.py ---
        criteria_refineLM = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1.19209e-07
        )

        # --- Pose estimator — same constructor args as real_multicam.py ---
        pose_estimator = PoseEstimator(
            framerate=self._fps,
            plotting=False,
            aruco_dict=aruco_dict,
            LMcriteria=criteria_refineLM,
            ransac=False,
            ransacTreshold=8,
        )

        # --- Load board points from .txt files (MARKER_MAPPER=True, OFFSET=False) ---
        try:
            target_points = np.loadtxt(self._calib_target, dtype=np.float32).reshape(-1, 4, 3)
            ref_points    = np.loadtxt(self._calib_ref,    dtype=np.float32).reshape(-1, 4, 3)
        except Exception as exc:
            print(f"[FalconTracker] board calibration load failed: {exc}")
            return

        target_board = cv2.aruco.Board(target_points, aruco_dict, np.arange(11))
        ref_board    = cv2.aruco.Board(ref_points,    aruco_dict, np.arange(11, 22))

        # --- Load camera calibration from JSON (same file as real_multicam.py) ---
        try:
            with open(self._calib_camera) as f:
                calib_entries = json.load(f)
            calib_by_id = {int(e['id']): e for e in calib_entries}
        except Exception as exc:
            print(f"[FalconTracker] camera calibration load failed: {exc}")
            return

        # Per-source camera matrices and dist coeffs
        cam_matrices = []
        dist_coeffs  = []
        fallback_id  = next(iter(calib_by_id))
        for cam_id in self._cam_ids:
            entry = calib_by_id.get(cam_id, calib_by_id[fallback_id])
            cam_matrices.append(np.array(entry['cameraMatrix'], dtype=np.float64))
            dist_coeffs.append(np.array(entry['distCoeffs'],   dtype=np.float64))

        # --- Open video captures ---
        caps = [cv2.VideoCapture(s) for s in self._sources]

        dt          = 1.0 / self._fps
        last_status = "NEVER_SEEN"
        n_cams      = len(caps)
        executor    = ThreadPoolExecutor(max_workers=n_cams)

        def _process_camera(i):
            cap = caps[i]
            ret, frame = cap.read()
            if not ret:
                if isinstance(self._sources[i], str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    return None

            corners, ids, _ = detector.detectMarkers(frame)
            try:
                pose, _ = pose_estimator.estimate_pose_board(
                    ref_board, target_board, corners, ids,
                    cam_matrices[i], dist_coeffs[i],
                )
                return np.asarray(pose).flatten() if pose is not None else None
            except Exception:
                return None

        final_pose = np.zeros((6, 1))

        try:
            while self._running:
                t_start = time.monotonic()

                results = list(executor.map(_process_camera, range(n_cams)))
                poses   = [r for r in results if r is not None]

                if poses:
                    poses_arr  = np.array(poses)          # (N, 6)
                    # Translation: simple mean (same as real_multicam.py)
                    final_pose[:3] = np.mean(poses_arr[:, :3], axis=0).reshape(3, 1)
                    # Rotation: anglesMean (same as real_multicam.py)
                    for j in range(3, 6):
                        final_pose[j] = _angles_mean(poses_arr[:, j])

                    if last_status != "SEEN":
                        last_status = "SEEN"
                        self.status_changed.emit("PointerToTracker", "SEEN")
                else:
                    if last_status == "SEEN":
                        last_status = "NOT_SEEN"
                        self.status_changed.emit("PointerToTracker", "NOT_SEEN")

                # Build 4×4 matrix — same as IGT block in real_multicam.py
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, :3] = R_scipy.from_euler(
                    'ZYX', final_pose[3:].ravel(), degrees=True
                ).as_matrix().T
                matrix[:3, 3] = final_pose[:3].flatten()
                self.transform_ready.emit("PointerToTracker", matrix)

                # Throttle to target fps
                sleep_time = dt - (time.monotonic() - t_start)
                deadline = time.monotonic() + max(sleep_time, 0)
                while self._running and time.monotonic() < deadline:
                    time.sleep(0.005)

        finally:
            executor.shutdown(wait=False)
            for cap in caps:
                cap.release()

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Public API  (matches MockIGTLClient interface)
# ---------------------------------------------------------------------------

class FalconTracker(QObject):
    """Multi-camera FALCON tracker using the real_multicam.py pipeline.

    Drop-in replacement for MockIGTLClient.

    Parameters
    ----------
    video_paths : list of str, optional
        Paths to pre-recorded video files (one per camera).
    camera_ids : list of int, optional
        Live OpenCV camera device indices.
    calib_ids : list of int, optional
        Calibration entry IDs (from the JSON) per source. Defaults to [1..N].
    calib_camera : str
        Path to the camera calibration JSON.
    calib_target : str
        Path to the target board .txt file.
    calib_ref : str
        Path to the reference board .txt file.
    fps : int
        Target processing rate (default 30).
    """

    transform_received  = Signal(str, object)
    tool_status_changed = Signal(str, str)
    connected           = Signal()
    disconnected        = Signal()

    def __init__(
        self,
        video_paths:   Optional[List[str]] = None,
        camera_ids:    Optional[List[int]] = None,
        calib_ids:     Optional[List[int]] = None,
        calib_camera:  str = _DEFAULT_CALIB_CAMERA,
        calib_target:  str = _DEFAULT_CALIB_TARGET,
        calib_ref:     str = _DEFAULT_CALIB_REF,
        fps:           int = 30,
        parent=None,
    ):
        super().__init__(parent)

        sources = video_paths if video_paths else (camera_ids or [])
        n = len(sources)
        if calib_ids is None:
            calib_ids = list(range(1, n + 1))

        self._sources      = sources
        self._cam_ids      = calib_ids[:n]
        self._calib_camera = calib_camera
        self._calib_target = calib_target
        self._calib_ref    = calib_ref
        self._fps          = fps

        self._thread: Optional[QThread]       = None
        self._worker: Optional[_FalconWorker] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self._sources:
            print("[FalconTracker] no sources configured — not starting")
            return

        self._worker = _FalconWorker(
            self._sources, self._cam_ids,
            self._calib_camera, self._calib_target, self._calib_ref,
            self._fps,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._worker.transform_ready.connect(self.transform_received)
        self._worker.status_changed.connect(self.tool_status_changed)
        self._thread.started.connect(self._worker.run)
        self._thread.start()
        self.connected.emit()

    def stop(self) -> None:
        if self._worker:
            self._worker.stop()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            if not self._thread.wait(5000):
                self._thread.terminate()
                self._thread.wait(1000)
        self._thread = None
        self._worker = None
        self.disconnected.emit()
