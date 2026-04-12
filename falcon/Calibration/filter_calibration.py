# # Add the parent directory to sys.path
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tracking', 'util')))

import cv2
import numpy as np
import dodecaBoard
from pose_estimation import pose_estimation
import matplotlib.pyplot as plt
import multiprocessing as mp
from pynput import keyboard
import pandas as pd

calibFile = 'calib_files/camera/currentCalibration.json'

poseEstimator = pose_estimation(framerate=60, plotting=False)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
arucoParams = cv2.aruco.DetectorParameters()

arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # Default CORNER_REFINE_NONE
arucoParams.cornerRefinementMaxIterations = 1000  # Default 30
arucoParams.cornerRefinementMinAccuracy = 0.001  # Default 0.1
arucoParams.adaptiveThreshWinSizeStep = 2  # Default 10
arucoParams.adaptiveThreshWinSizeMax = 15  # Default 23
arucoParams.adaptiveThreshConstant = 8  # Default 7
# arucoParams.minCornerDistanceRate = 0.01
detector = cv2.aruco.ArucoDetector(poseEstimator.aruco_dict, arucoParams)

m = 33.2/2 # half of marker length (currently in mm)

# Single marker board
# board_points = np.array([[[-m, m, 0],[m, m, 0],[m, -m, 0],[-m, -m, 0]]],dtype=np.float32)

# ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([0]))
# target_board = cv2.aruco.Board(board_points, aruco_dict, np.array([1]))

# Dodecahedron board
target_marker_size = 24  # dodecahedron edge length in mm
target_pentagon_size = 27.5
ref_marker_size = 35  # dodecahedron edge length in mm
ref_pentagon_size = 40
targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (0, 0, 253), 'centre')
refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size, (0, 0, 109), 'centre')
target_board = cv2.aruco.Board(targetPoints, aruco_dict, np.arange(11))
ref_board = cv2.aruco.Board(refPoints, aruco_dict, np.arange(11,22))


'''
===========================================================================
Initialize covariance matrices Q and R for noise modelling in Kalman filter
===========================================================================

For one camera there is only a covariance matrix R modelling measurement noise.
The matrix R consists of the mean variance of each pose element across a specified number of static positions.

For multiple cameras, there is a covariance matrix Q modelling process noise and a covariance matrix R' modelling measurement noise.
The matrix Q consists of the mean covariance R across all cameras.
The matrix R' is a diagonal matrix synthesized from the covariance matrices R of a set of cameras n, and therefore has shape (6*n, 6*n).
'''

def compute_R(cap, cameraMatrix, distCoeffs):
    global covariances
    global norms
    
    elements = np.zeros([6, num_frames], dtype=np.float64)
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Estimate pose
        try:
            frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
            corners, ids, _ = detector.detectMarkers(frame)
            pose, _ = poseEstimator.estimate_pose_board(ref_board, target_board, corners, ids, cameraMatrix)
            elements[:, i] = pose.flatten()

        except:
            print(f"Camera {cam}: Failed to detect markers in frame {i+1}/{num_frames}")
    
    # Calculate variance across each pose element (row-wise)
    variances = np.var(elements, axis=1, dtype=np.float64)
    
    # Create diagonal matrix R
    R = np.diag(variances)
    covariances[cam].append(R)

    print(f"Camera {cam}: The covariance matrix R at position {len(covariances[cam])} is:\n {R}")
    
    # Calculate and store Frobenius norm
    if xyz_norm:
        norm = np.linalg.norm(R[:3, :3], 'fro')  # Use only the top left 3x3 block
        print(f"Camera {cam}: The XYZ Frobenius norm of R at position {len(covariances[cam])} is: {norm}\n")
    else:
        norm = np.linalg.norm(R, 'fro')  # Use the full matrix
        print(f"Camera {cam}: The Frobenius norm of R at position {len(covariances[cam])} is: {norm}\n")

    norms[cam].append(norm)

def compute_R_mean():
    for cam in cams:
        # Filter out zero covariance matrices
        non_zero_covariances = [cov for cov in covariances[cam] if np.any(cov)]
        
        if non_zero_covariances:
            # Compute the mean covariance matrix
            stacked_covariances = np.stack(non_zero_covariances)
            R_mean = np.mean(stacked_covariances, axis=0)
            mean_covariances[cam].append(R_mean)
            
            if xyz_norm:
                R_mean_norm = np.linalg.norm(R_mean[:3, :3], 'fro')  # Use only the top left 3x3 block
            else:
                R_mean_norm = np.linalg.norm(R_mean, 'fro')  # Use the full matrix
            
            mean_norms.append(R_mean_norm)
                
            print(f"Camera {cam}: The mean covariance matrix R is:\n{R_mean}")
            print(f"Camera {cam}: The Frobenius norm of the mean R is: {R_mean_norm}\n")
        else:
            print(f"Camera {cam}: No non-zero covariance matrices to compute the mean.")
    
    # Store the list with the mean covariance matrix R for each camera
    np.save(f'R', mean_covariances)
    print(f"The mean covariance matrix R for each camera was saved to 'R.npy\n'")

def compute_Q():
    # Compute the overall mean covariance matrix Q
    all_mean_covariances = np.stack([mean_covariances[cam] for cam in cams])
    Q = np.mean(all_mean_covariances, axis=0)
    print(f"The mean covariance matrix Q is:\n{Q}")

    # Store the matrix R_mean
    np.save(f'Q.npy', Q)
    print(f"The mean covariance matrix Q was saved to 'Q.npy'")


def plot_pos_norms():
    for cam in cams:
        positions = range(1, len(norms[cam]) + 1)
        plt.figure()
        bars = plt.bar(positions, norms[cam])
        plt.title(f'Frobenius Norms of R Matrices at Each Position for Camera {cam}')
        plt.xlabel('Position')
        plt.ylabel('Norm [mm]')
        plt.xticks(positions)
        for bar, norm in zip(bars, norms[cam]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{norm:.2f}', ha='center', va='bottom')
        plt.grid(False)  # Disable grid
        plt.show()

def plot_cam_norms():
    plt.figure()
    bars = plt.bar(range(len(cams)), mean_norms)
    plt.title('Mean Frobenius Norm of R Matrices for Each Camera')
    plt.xlabel('Camera')
    plt.ylabel('Mean Norm [mm]')
    plt.xticks(range(len(cams)), [f'{cam}' for cam in cams])
    for bar, mean_norm in zip(bars, mean_norms):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{mean_norm:.2f}', ha='center', va='bottom')
    plt.grid(False)  # Disable grid
    plt.show()

def plot_r_norm_matrix():
    # Create a matrix to store the norms of each camera at each position
    r_norm_matrix = np.zeros((len(cams), num_positions))

    for i, cam in enumerate(cams):
        for j in range(num_positions):
            r_norm_matrix[i, j] = norms[cam][j]

    plt.figure()
    plt.imshow(r_norm_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Frobenius Norms of R Matrices for Each Camera and Position')
    plt.xlabel('Position')
    plt.ylabel('Camera')
    plt.xticks(range(num_positions), range(1, num_positions + 1))
    plt.yticks(range(len(cams)), [f'{cam}' for cam in cams])
    for i in range(len(cams)):
        for j in range(num_positions):
            plt.text(j, i, f'{r_norm_matrix[i, j]:.2f}', ha='center', va='center', color='black')
    plt.show()

def plot_diff_matrix():
    for cam in cams:
        norm_differences = np.zeros((num_positions, num_positions))

        for i in range(num_positions):
            for j in range(num_positions):
                norm_differences[i, j] = abs(norms[cam][i] - norms[cam][j])

        plt.figure()
        plt.imshow(norm_differences, cmap='viridis')
        plt.colorbar()
        plt.title(f'Absolute Differences of Norms Between Positions for Camera {cam}')
        plt.xlabel('Position Index')
        plt.ylabel('Position Index')
        plt.xticks(range(num_positions), range(1, num_positions + 1))
        plt.yticks(range(num_positions), range(1, num_positions + 1))
        for i in range(num_positions):
            for j in range(num_positions):
                plt.text(j, i, f'{norm_differences[i, j]:.2f}', ha='center', va='center', color='black')
        plt.show()

def plot_mean_diff_matrix():
    all_norm_differences = np.zeros((len(cams), num_positions, num_positions))

    # Compute the norm differences for each camera
    for cam_idx, cam in enumerate(cams):
        norm_differences = np.zeros((num_positions, num_positions))

        for i in range(num_positions):
            for j in range(num_positions):
                norm_differences[i, j] = abs(norms[cam][i] - norms[cam][j])

        all_norm_differences[cam_idx] = norm_differences

    # Calculate the mean difference matrix across all cameras
    mean_norm_differences = np.mean(all_norm_differences, axis=0)

    # Plot the mean difference matrix
    plt.figure()
    plt.imshow(mean_norm_differences, cmap='viridis')
    plt.colorbar()
    plt.title('Mean Absolute Differences of Norms Between Positions')
    plt.xlabel('Position Index')
    plt.ylabel('Position Index')
    plt.xticks(range(num_positions), range(1, num_positions + 1))
    plt.yticks(range(num_positions), range(1, num_positions + 1))
    for i in range(num_positions):
        for j in range(num_positions):
            plt.text(j, i, f'{mean_norm_differences[i, j]:.2f}', ha='center', va='center', color='black')
    plt.show()

def capture_frames(cam, capture_event, stop_event, parent_conn, cameraMatrix, distCoeffs):
    cap = cv2.VideoCapture(f"udpsrc address=192.168.5.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Cannot open camera {cam}.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
        corners, ids, _ = detector.detectMarkers(frame)
        overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow(f'Camera {cam}', overlayImg)

        if capture_event.is_set():
            compute_R(cap, cameraMatrix, distCoeffs)
            barrier.wait()
            capture_event.clear()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    
    cap.release()
    cv2.destroyAllWindows()

def on_press(key):
    global capture_event, stop_event
    try:
        if key.char == 'c':
            capture_event.set()
        elif key.char == 'q':
            stop_event.set()
    except AttributeError:
        pass

if __name__ == "__main__":
    cams = [1, 2, 3, 4, 5]  # Camera IDs that correspond to label on pi and port number 500X
    num_frames = 60
    num_positions = 8
    xyz_norm = False  # Flag to calculate norms based only on the x, y, z pose elements
    processes = []
    parent_conns = []
    capture_event = mp.Event()
    stop_event = mp.Event()
    barrier = mp.Barrier(len(processes)+1)
    manager = mp.Manager()
    covariances = manager.dict({cam: manager.list() for cam in cams})
    norms = manager.dict({cam: manager.list() for cam in cams})
    mean_covariances = {cam: [] for cam in cams} # Mean R matrix for each camera
    mean_norms = [] # Mean R norm for each camera
    calibData = pd.read_json(calibFile, orient='index')
    calibData = calibData.applymap(np.array)
    calibData = calibData.replace(np.nan, None)

    for cam in cams:
        cameraMatrix = cv2.Mat(calibData.at[str(cam), "cameraMatrix"])
        distCoeffs = cv2.Mat(calibData.at[str(cam), "distCoeffs"])
        parent_conn, child_conn = mp.Pipe()
        process = mp.Process(target=capture_frames, args=(cam, capture_event, stop_event, child_conn, cameraMatrix, distCoeffs))
        process.start()
        processes.append(process)
        parent_conns.append(parent_conn)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        while True:
            if stop_event.is_set():
                print(covariances)
                break
            if all(len(covariances[cam]) == num_positions for cam in cams):
                compute_R_mean()
                compute_Q()
                plot_cam_norms()
                plot_r_norm_matrix()
                break
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        for process in processes:
            process.join()
        listener.stop()