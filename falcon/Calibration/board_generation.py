# # Add the parent directory to sys.path
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tracking', 'util')))

# import cv2
# import numpy as np
# import dodecaBoard
# from pose_estimation import pose_estimation
# import matplotlib.pyplot as plt
# import multiprocessing as mp
# from pynput import keyboard
# import pandas as pd

# calibFile = 'currentCalibration.json'

# poseEstimator = pose_estimation(framerate=60, plotting=False)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
# arucoParams = cv2.aruco.DetectorParameters()

# arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # Default CORNER_REFINE_NONE
# arucoParams.cornerRefinementMaxIterations = 1000  # Default 30
# arucoParams.cornerRefinementMinAccuracy = 0.001  # Default 0.1
# arucoParams.adaptiveThreshWinSizeStep = 2  # Default 10
# arucoParams.adaptiveThreshWinSizeMax = 15  # Default 23
# arucoParams.adaptiveThreshConstant = 8  # Default 7
# # arucoParams.minCornerDistanceRate = 0.01
# detector = cv2.aruco.ArucoDetector(poseEstimator.aruco_dict, arucoParams)

# m = 33.2/2 # half of marker length (currently in mm)

# # Single marker board
# # board_points = np.array([[[-m, m, 0],[m, m, 0],[m, -m, 0],[-m, -m, 0]]],dtype=np.float32)

# # ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([0]))
# # target_board = cv2.aruco.Board(board_points, aruco_dict, np.array([1]))

# # Dodecahedron board
# target_marker_size = 24  # dodecahedron edge length in mm
# target_pentagon_size = 27.5
# ref_marker_size = 35  # dodecahedron edge length in mm
# ref_pentagon_size = 40
# targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (0, 0, 253), 'centre')
# refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size, (0, 0, 109), 'centre')
# target_board = cv2.aruco.Board(targetPoints, aruco_dict, np.arange(11))
# ref_board = cv2.aruco.Board(refPoints, aruco_dict, np.arange(11,22))
# calibFile = 'currentCalibration.json'

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

# def capture_frames(cam, capture_event, stop_event, parent_conn, cameraMatrix, distCoeffs):
#     cap = cv2.VideoCapture(f"udpsrc address=192.168.5.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
#     if not cap.isOpened():
#         print(f"Cannot open camera {cam}.")
#         return

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = None

#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
#         corners, ids, _ = detector.detectMarkers(frame)
#         overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
#         cv2.imshow(f'Camera {cam}', overlayImg)

#         if capture_event.is_set():
#             if out is None:
#                 out = cv2.VideoWriter(f'cam{cam}.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
#                 out.write(frame)
#                 cv2.putText(frame, 'Recording', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#             else:
#                 if out is not None:
#                     out.release()
#                     out = None

#             barrier.wait()
#             capture_event.clear()

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             stop_event.set()
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# def on_press(key):
#     global capture_event, stop_event
#     try:
#         if key.char == 'c':
#             capture_event.set()
#         elif key.char == 'q':
#             stop_event.set()
#     except AttributeError:
#         pass

# if __name__ == "__main__":
#     cams = [1]  # Camera IDs that correspond to label on pi and port number 500X
#     processes = []
#     parent_conns = []
#     capture_event = mp.Event()
#     stop_event = mp.Event()
#     barrier = mp.Barrier(len(processes)+1)
#     manager = mp.Manager()
#     calibData = pd.read_json(calibFile, orient='index')
#     calibData = calibData.applymap(np.array)
#     calibData = calibData.replace(np.nan, None)

#     for cam in cams:
#         cameraMatrix = cv2.Mat(calibData.at[str(cam), "cameraMatrix"])
#         distCoeffs = cv2.Mat(calibData.at[str(cam), "distCoeffs"])
#         parent_conn, child_conn = mp.Pipe()
#         process = mp.Process(target=capture_frames, args=(cam, capture_event, stop_event, child_conn, cameraMatrix, distCoeffs))
#         process.start()
#         processes.append(process)
#         parent_conns.append(parent_conn)
    
#     listener = keyboard.Listener(on_press=on_press)
#     listener.start()

#     try:
#         while True:
#             if stop_event.is_set():
#                 print("Board generation aborted.")
#                 break
#             # if all videos are collected:
#             #     compute all board files
#             #     if more than one files are computed:
#             #         use the one with the smallest reprojection error
#             #     else: use the one that is computed
#             #     print("Stored board generated using camera {camera with lowest error}. Error: {error}")
#             #     break
#     except KeyboardInterrupt:
#         stop_event.set()
#     finally:
#         for process in processes:
#             process.join()
#         listener.stop()

import cv2
import os

def capture_video(cam, output_file, fps=60):
    # Open the camera
    cap = cv2.VideoCapture(f"udpsrc address=192.168.5.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {cam}")
        return
    
    # Read the first frame to get the dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        cap.release()
        return
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    print(f"{frame_height, frame_width}")
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print(f"Error: Cannot open video writer with file {output_file}")
        cap.release()
        return
    
    print(f"Recording... Press 'q' to stop. Saving to {os.path.abspath(output_file)}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Write the frame to the output file
        out.write(frame)
        
        # Display the frame
        cv2.imshow('frame', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    cam = input("Enter the camera ID or pipeline string: ")
    output_file = input("Enter the output file name (with .avi extension): ")
    
    capture_video(cam, output_file)