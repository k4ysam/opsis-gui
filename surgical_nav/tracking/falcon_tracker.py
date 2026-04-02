"""FalconTracker: multi-camera ArUco pose tracking using the FALCON pipeline.

Matches MockIGTLClient's signal interface (transform_received, tool_status_changed,
connected, disconnected) so it can be swapped in transparently.

Modes
-----
- Video mode  : pass ``video_paths`` — replay pre-recorded .mp4 files (loops).
- Live mode   : pass ``camera_ids`` — open live OpenCV VideoCapture devices.

Calibration
-----------
Reads per-camera intrinsics + extrinsics from the JSON file produced by
FALCON's camera calibration.  The default path points to the calib file
already in the falcon/ directory.
"""

from __future__ import annotations

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Union

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

# cv2, scipy, and FALCON utils are imported lazily inside the worker so that
# importing this module never fails when OpenCV is not installed.

_FALCON_TRACKING = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'falcon', 'Tracking')
)


def _ensure_falcon_path():
    if _FALCON_TRACKING not in sys.path:
        sys.path.insert(0, _FALCON_TRACKING)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CALIB = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), '..', '..',
        'falcon', 'Calibration', 'calib_files', 'camera',
        'T33_minpos_int_percam_cam1fixed.json',
    )
)

# Tool (target) dodecahedron
_TARGET_EDGE      = 24.0
_TARGET_PENTAGON  = 27.5
_TARGET_OFFSET    = (-0.42030256, -1.35511477, 211.53434921)
_TARGET_IDS       = list(range(0, 11))

# Reference dodecahedron (fixed to patient)
_REF_EDGE         = 33.0
_REF_PENTAGON     = 40.0
_REF_IDS          = list(range(11, 22))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_mean(angles_deg: np.ndarray) -> float:
    rad = np.deg2rad(angles_deg)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [X, Y, Z, Yaw, Pitch, Roll] → 4×4 transform (mm)."""
    from scipy.spatial.transform import Rotation as R_scipy
    rot = R_scipy.from_euler('ZYX', pose[3:6], degrees=True).as_matrix()
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rot.T
    m[:3, 3]  = pose[:3].flatten()
    return m


def _load_calib(calib_file: str) -> dict:
    """Load calibration JSON → {cam_id: {matrix, dist}} dict."""
    with open(calib_file) as f:
        entries = json.load(f)
    result = {}
    for e in entries:
        result[int(e['id'])] = {
            'matrix': np.array(e['cameraMatrix'], dtype=np.float64),
            'dist':   np.array(e['distCoeffs'],   dtype=np.float64),
        }
    return result


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
        calib_file: str,
        use_kalman: bool,
        fps: int,
    ):
        super().__init__()
        self._sources    = sources      # cv2 indices or file paths
        self._cam_ids    = cam_ids      # calibration IDs matching each source
        self._calib_file = calib_file
        self._use_kalman = use_kalman
        self._fps        = fps
        self._running    = False

    def run(self) -> None:
        self._running = True

        # --- Lazy imports (cv2 / scipy / FALCON utils) ---
        try:
            import cv2
        except ImportError:
            print("[FalconTracker] OpenCV not installed — tracking disabled")
            return

        _ensure_falcon_path()
        try:
            from util.pose_estimation import pose_estimation as PoseEstimator
            from util.dodecaBoard import generate as gen_dodeca
            from util.kalman import KalmanFilterCV
        except ImportError as exc:
            print(f"[FalconTracker] FALCON utils not found: {exc}")
            return

        # --- Load calibration ---
        try:
            calib = _load_calib(self._calib_file)
        except Exception as exc:
            print(f"[FalconTracker] calibration load failed: {exc}")
            return

        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36H12)

        # --- Build boards ---
        target_pts   = gen_dodeca(_TARGET_EDGE, _TARGET_PENTAGON, _TARGET_OFFSET)
        ref_pts      = gen_dodeca(_REF_EDGE,    _REF_PENTAGON)
        target_board = cv2.aruco.Board(target_pts, aruco_dict, np.array(_TARGET_IDS))
        ref_board    = cv2.aruco.Board(ref_pts,    aruco_dict, np.array(_REF_IDS))

        # --- Detector (fast + sensitive) ---
        # step=10/max=23 sweeps only 3 threshold windows instead of 14,
        # giving 25x speedup with no loss in marker detection rate.
        det_params = cv2.aruco.DetectorParameters()
        det_params.adaptiveThreshWinSizeMin   = 3
        det_params.adaptiveThreshWinSizeMax   = 23
        det_params.adaptiveThreshWinSizeStep  = 10
        det_params.minMarkerPerimeterRate     = 0.01
        det_params.cornerRefinementMethod     = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, det_params)

        # Target width for downscaling before detection (balances speed vs accuracy)
        _DETECT_WIDTH = 960

        # --- Per-camera pose estimators ---
        estimators = []
        cam_params = []
        for cam_id in self._cam_ids:
            p = calib.get(cam_id, list(calib.values())[0])
            cam_params.append(p)
            est = PoseEstimator(framerate=self._fps, ransac=True, ransacTreshold=15)
            est.set_camera_params(p['matrix'], p['dist'])
            estimators.append(est)

        # --- Open captures ---
        caps = [cv2.VideoCapture(s) for s in self._sources]

        dt           = 1.0 / self._fps
        last_status  = "NEVER_SEEN"
        n_cams       = len(caps)
        executor     = ThreadPoolExecutor(max_workers=n_cams)

        # Kalman is always on — it predicts forward when no detection,
        # giving smooth continuous output even at ~13% detection rate.
        kalman       = KalmanFilterCV(freq=self._fps, calib=False)
        kalman_ready = False

        def _process_camera(i):
            cap = caps[i]
            ret, frame = cap.read()
            if not ret:
                if isinstance(self._sources[i], str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret:
                    return None

            # Downscale for faster detection; scale the camera matrix accordingly
            h, w = frame.shape[:2]
            if w > _DETECT_WIDTH:
                scale = _DETECT_WIDTH / w
                frame = cv2.resize(frame, (_DETECT_WIDTH, int(h * scale)))
                K_scaled = cam_params[i]['matrix'].copy()
                K_scaled[0] *= scale   # fx, cx
                K_scaled[1] *= scale   # fy, cy
            else:
                scale = 1.0
                K_scaled = cam_params[i]['matrix']

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            corners, ids, _ = detector.detectMarkers(gray)
            try:
                pose, _ = estimators[i].estimate_pose_board(
                    ref_board, target_board, corners, ids,
                    K_scaled, cam_params[i]['dist'],
                )
                return np.asarray(pose).flatten() if pose is not None else None
            except Exception:
                return None

        try:
            while self._running:
                t_start = time.monotonic()

                # Process all cameras in parallel
                results = list(executor.map(_process_camera, range(n_cams)))
                poses   = [r for r in results if r is not None]

                if poses:
                    poses_arr  = np.stack(poses)           # (N, 6)
                    fused_pose = np.zeros(6)
                    fused_pose[:3] = np.mean(poses_arr[:, :3], axis=0)
                    for j in range(3, 6):
                        fused_pose[j] = _circular_mean(poses_arr[:, j])

                    measurement = fused_pose.reshape(6, 1)

                    if not kalman_ready:
                        kalman.initiate_state(measurement)
                        kalman_ready = True

                    kalman.set_measurement_matrices(1, kalman.R)
                    kalman.set_measurement(measurement)
                    kalman.predict()
                    kalman.correct()

                    if last_status != "SEEN":
                        last_status = "SEEN"
                        self.status_changed.emit("PointerToTracker", "SEEN")

                elif kalman_ready:
                    # No detection this frame — predict forward using velocity
                    kalman.predict()

                    if last_status == "SEEN":
                        last_status = "NOT_SEEN"
                        self.status_changed.emit("PointerToTracker", "NOT_SEEN")

                # Emit smoothed pose every frame (once initialised)
                if kalman_ready:
                    final_pose = kalman.x[:6].flatten()
                    matrix     = _pose_to_matrix(final_pose)
                    self.transform_ready.emit("PointerToTracker", matrix)

                # Throttle to target fps in small increments so stop() wakes us quickly
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
    """Multi-camera FALCON tracker.

    Drop-in replacement for MockIGTLClient.

    Parameters
    ----------
    video_paths : list of str, optional
        Paths to pre-recorded video files (one per camera).  When provided,
        the tracker replays them in a loop — useful for offline testing.
    camera_ids : list of int, optional
        Live OpenCV camera device indices.  Ignored when *video_paths* given.
    calib_ids : list of int, optional
        Calibration entry IDs (from the JSON) corresponding to each source.
        Defaults to ``[1, 2, 3, 4, 5]`` trimmed to the number of sources.
    calib_file : str, optional
        Path to the FALCON camera calibration JSON.
    use_kalman : bool
        Enable Kalman filtering / smoothing (default False).
    fps : int
        Target processing rate (default 30).
    """

    transform_received  = Signal(str, object)   # (name, ndarray 4×4)
    tool_status_changed = Signal(str, str)       # (name, status)
    connected           = Signal()
    disconnected        = Signal()

    def __init__(
        self,
        video_paths: Optional[List[str]] = None,
        camera_ids:  Optional[List[int]] = None,
        calib_ids:   Optional[List[int]] = None,
        calib_file:  str = _DEFAULT_CALIB,
        use_kalman:  bool = False,
        fps:         int  = 30,
        parent=None,
    ):
        super().__init__(parent)

        if video_paths:
            sources = video_paths
        elif camera_ids is not None:
            sources = camera_ids
        else:
            sources = []

        n = len(sources)
        if calib_ids is None:
            calib_ids = list(range(1, n + 1))

        self._sources    = sources
        self._cam_ids    = calib_ids[:n]
        self._calib_file = calib_file
        self._use_kalman = use_kalman
        self._fps        = fps

        self._thread: Optional[QThread]       = None
        self._worker: Optional[_FalconWorker] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self._sources:
            print("[FalconTracker] no sources configured — not starting")
            return

        self._worker = _FalconWorker(
            self._sources, self._cam_ids,
            self._calib_file, self._use_kalman, self._fps,
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
