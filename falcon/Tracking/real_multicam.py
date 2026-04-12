import util.dodecaBoard as dodecaBoard
from util.pose_estimation import pose_estimation
from util.kalman import KalmanFilterCV
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import multiprocessing as mp
import time
import pyigtl
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import itertools
from util.landmark_registration import register, apply_transform
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats.mstats import hdquantiles

### PRESS Q ON ANY CAMERA WINDOW TO QUIT ###

RATE = 100
PRERECORDED = True
STEREO = False
KALMAN = False
ADAPTIVE = False
REJECT_OUTLIER_TAGS = False
REJECT_THRESH = 8  # reprojection error rejection threshold, default 8
REJECT_OUTLIER_CAMS = False
OFFSET = False
MARKER_MAPPER = True  # this flag currently just uses the calibFiles for mono, not the mapperFiles
STEREO_REPROJECT = False
STEREO_TRIANGULATE_PAIRS = False
STEREO_TRIANGULATE_ALL = True
PLOTTING = False
PLOTTING_3D = False
CROP = True
CROP_PADDING = 100
ROS = False
IGT = True
IGT_Port = 18995
DISPLAY = True
cams = [1,2,3,4] # Camera IDs that correspond to label on pi and port number 500X
toolOffset = (-0.37746276, -0.50223948, -174.26308552)
calibFile_camera = '../Calibration/calib_files/camera/T33_minpos_int_percam_cam1fixed.json'
# calibFile_camera = '../Calibration/calib_files/camera/T33int_T53ext.json'
calibFile_target = '../Calibration/calib_files/tool/T33_minpos_int_percam_cam1fixed_target.txt'
calibFile_ref = '../Calibration/calib_files/tool/T33_minpos_int_percam_cam1fixed_ref.txt'
# calibFile_target = '../Calibration/calib_files/tool/target_T33int_T53ext_minreproj.txt'
# calibFile_ref = '../Calibration/calib_files/tool/ref_T33int_T53ext_minreproj.txt'
# mapperFile_target = '../Calibration/calib_files/marker_mapper/target_map_T33_3.yml'
# mapperFile_ref = '../Calibration/calib_files/marker_mapper/ref_map_T33_3.yml'
RFile_mono = '../Calibration/calib_files/filter/R_30x30_mono.txt'
QFile_stereo = '../Calibration/calib_files/filter/Q_matrix_stereo.txt'
videoFiles = 'video/GT01_RT1'  # Directory that contains a {cam}.{videoExt} video file for each camqera ID being used
videoExt = 'mp4' # extension of provided video files
imageSize = 1456, 1088
cams_all = [1,2,3,4,5]  # list of all possible cam IDs

if ROS:
    import rospy
    from geometry_msgs.msg import Pose
    from gazebo_msgs.msg import ModelState 

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5) # Works with marker mapper
criteria_refineLM = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 200, 1.19209e-07)  # Default count = 20, epsilon = 1.19209e-07
poseEstimator = pose_estimation(framerate=RATE, plotting=False, aruco_dict=aruco_dict, LMcriteria=criteria_refineLM, ransac=REJECT_OUTLIER_TAGS, ransacTreshold=REJECT_THRESH)

arucoParams = cv2.aruco.DetectorParameters()
arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # Default CORNER_REFINE_NONE
arucoParams.cornerRefinementMaxIterations = 1000  # Default 30
arucoParams.cornerRefinementMinAccuracy = 0.001  # Default 0.1
arucoParams.adaptiveThreshWinSizeStep = 2  # Default 10
arucoParams.adaptiveThreshWinSizeMax = 30  # Default 23
arucoParams.adaptiveThreshConstant = 8  # Default 7
# arucoParams.aprilTagMinWhiteBlackDiff = 1  # Default 7
# arucoParams.polygonalApproxAccuracyRate = 0.01  # Default 0.03

detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

# m = 33.2/2 # half of marker length (currently in mm)

# # Single marker board
# board_points = np.array([[[-m, m, 0],[m, m, 0],[m, -m, 0],[-m, -m, 0]]],dtype=np.float32)

# ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([0]))
# ref_board = cv2.aruco.Board(board_points, aruco_dict, np.array([1]))

# Dodecahedron board
target_marker_size = 24  # dodecahedron edge length in mm
target_pentagon_size = 27.5
ref_marker_size = 33  # dodecahedron edge length in mm
ref_pentagon_size = 40

# Combined function to read and parse the marker map YAML file
def load_markerMapper(file_path, offset=(0.0, 0.0, 0.0)):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Skip lines starting with '%', in this case the directive
    yaml_content = ''.join(line for line in lines if not line.strip().startswith('%'))
    data = yaml.safe_load(yaml_content)
    
    markers = data.get('aruco_bc_markers', [])
    
    if not markers:
        raise ValueError("No markers found in the YAML file.")
    
    # Extract the top marker's corners
    top_marker = markers[0]
    corners_top_marker = np.array(top_marker['corners'], dtype=np.float32)
    center_top_marker = np.mean(corners_top_marker, axis=0)
    
    # Calculate the normal vector of the top marker plane
    v1 = corners_top_marker[1] - corners_top_marker[0]
    v2 = corners_top_marker[2] - corners_top_marker[0]
    normal_top_marker = np.cross(v1, v2)
    normal_top_marker /= np.linalg.norm(normal_top_marker)
    normal_top_marker *= -1 # Invert normal so that is faces outwards from board center

    # Create a rotation matrix to align the normal with the z-axis
    z_axis = np.array([0, 0, 1], dtype=np.float32)
    axis = np.cross(normal_top_marker, z_axis)
    angle = np.arccos(np.clip(np.dot(normal_top_marker, z_axis), -1.0, 1.0))

    # Check if the normal is already aligned with the z-axis
    if np.linalg.norm(axis) != 0:
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        
        Rot = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    else:
        Rot = np.eye(3)

    parsed_markers = []
    for marker in markers:
        if 'corners' in marker:
            corners = np.array(marker['corners'], dtype=np.float32)
            # Translate and rotate corners
            translated_corners = corners - center_top_marker
            rotated_corners = np.dot(translated_corners, Rot.T)  # Apply rotation
            rotated_corners = rotated_corners[[2, 3, 0, 1], :] * 1000 # Flip marker orientation and go from m to mm
            offset_corners = rotated_corners - np.array(offset, dtype=np.float32) # Apply offset
            parsed_markers.append(offset_corners)
            # parsed_markers.append(corners[[2, 3, 0, 1], :] * 1000)

    # Convert parsed_markers to a numpy array of the same type
    return np.array(parsed_markers, dtype=np.float32)

if OFFSET and not MARKER_MAPPER:
    targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, toolOffset, 'centre')
    refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size)

elif OFFSET and MARKER_MAPPER:
    targetPoints = load_markerMapper(mapperFile_target, toolOffset)
    refPoints = load_markerMapper(mapperFile_ref)

elif MARKER_MAPPER and not OFFSET:
    # targetPoints = load_markerMapper(mapperFile_target)
    # refPoints = load_markerMapper(mapperFile_ref)
    targetPoints = np.loadtxt(calibFile_target, dtype=np.float32).reshape(-1,4,3)
    refPoints = np.loadtxt(calibFile_ref, dtype=np.float32).reshape(-1,4,3)

# if OFFSET and not MARKER_MAPPER:
#     # targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (-0.314668, 0.815797, 217.424), 'centre')
#     # targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (0,0, 214.424), 'centre')
#     # targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (5.90845, 1.16099, 221.491), 'centre')
#     # targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (-0.53453551, -0.18892355, 223.38273973), 'centre')
#     # targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (-0.42030256, -1.35511477, 211.53434921), 'centre')
#     targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size, (-0.46397128, -0.19475517, 213.41259916), 'centre')
#     refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size)

# elif OFFSET and MARKER_MAPPER:
#     # targetPoints = load_markerMapper('../Calibration/calib_files/marker_mapper/target_map_tag36h11_highres.yml', (0.53453551, 0.18892355, -284.62613973)) #-0.618269, -0.384169, -285.049
#     # targetPoints = load_markerMapper('../Calibration/calib_files/marker_mapper/target_map_tag36h11_highres.yml', (0,0, -275.6674)) #-0.618269, -0.384169, -285.049
#     # targetPoints = load_markerMapper('../Calibration/calib_files/marker_mapper/target_map_tag36h11_highres.yml', (-0.27902203, -1.12389910, -273.73267133))
#     targetPoints = load_markerMapper('../Calibration/calib_files/marker_mapper/target_map_T7.yml', (1.71724532, -2.30901155, -279.96302979))
#     refPoints = load_markerMapper('../Calibration/calib_files/marker_mapper/ref_map_T7.yml')

# elif MARKER_MAPPER and not OFFSET:
#     targetPoints = load_markerMapper('../Calibration/calib_files/marker_mapper/target_map_T7.yml')
#     refPoints = load_markerMapper('../Calibration/calib_files/marker_mapper/ref_map_T7.yml')
#     print(refPoints)

elif STEREO: 
    targetPoints = np.loadtxt(calibFile_target, dtype=np.float32).reshape(-1,4,3)
    refPoints = np.loadtxt(calibFile_ref, dtype=np.float32).reshape(-1,4,3)
else: 
    targetPoints = dodecaBoard.generate(target_marker_size, target_pentagon_size)
    refPoints = dodecaBoard.generate(ref_marker_size, ref_pentagon_size)

targetTagCorners = np.array([[-target_marker_size/2, target_marker_size/2, 0], 
                             [target_marker_size/2, target_marker_size/2, 0],
                             [target_marker_size/2, -target_marker_size/2, 0],
                             [-target_marker_size/2, -target_marker_size/2, 0]], dtype=np.float32)
refTagCorners = np.array([[-ref_marker_size/2, ref_marker_size/2, 0],
                          [ref_marker_size/2, ref_marker_size/2, 0],
                          [ref_marker_size/2, -ref_marker_size/2, 0],
                          [-ref_marker_size/2, -ref_marker_size/2, 0]], dtype=np.float32)
tagPoints = np.vstack([targetPoints, refPoints])

# print(f'Target points:\n{targetPoints}\n')
# print(f'Reference points:\n{refPoints}')

target_board = cv2.aruco.Board(targetPoints, aruco_dict, np.arange(11))
ref_board = cv2.aruco.Board(refPoints, aruco_dict, np.arange(11,22))

N_tagIds = len(target_board.getIds()) + len(ref_board.getIds())
N_cams = len(cams)

def terminate_processes(processes):
    for process in processes:
        process.terminate()
    # for process in processes:
    #     process.join()

def runCam(cam, cameraMatrix, distCoeffs, childConn, stopEvent, barrier):

    if PRERECORDED: cap = cv2.VideoCapture(f'{videoFiles}/{cam}.{videoExt}')
    else: cap = cv2.VideoCapture(f"udpsrc address=192.168.3.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print(f"Cannot open camera {cam}.")
        return
    
    # Read the first frame to get the dimensions
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read frame of cam {cam}")
        cap.release()
        return
    
    frame_height, frame_width = frame.shape[:2]
    imageSize = frame_width, frame_height
    # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, alpha=1, newImgSize=imageSize)
    
    if CROP: found = False
    
    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        barrier.wait()
        ret, frame = cap.read()  # ret is True if frame is read correctly
        if not ret:
            if PRERECORDED and not stopEvent.is_set():
                print("Prerecorded video finished.")
                stopEvent.set()
            elif not PRERECORDED:
                print(f"Can't receive frame from camera {cam}.")
            continue
        
        # frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
        # frame = cv2.undistort(frame, cameraMatrix, distCoeffs, newCameraMatrix=newCameraMatrix)
        
        if CROP and found:
            ids = []
            corners = []
            frame_target = frame[ymin_target:ymax_target, xmin_target:xmax_target]
            frame_ref = frame[ymin_ref:ymax_ref, xmin_ref:xmax_ref]
            
            corners_target, ids_target, rejected_target = detector.detectMarkers(frame_target)
            corners_target, ids_target, _, _ = detector.refineDetectedMarkers(frame_target, target_board, corners_target, ids_target, rejected_target, cameraMatrix, distCoeffs)
            
            if ids_target is not None:
                corners_target = np.array(corners_target) + (xmin_target, ymin_target)
                corners.append(corners_target)
                ids.append(ids_target)
                
            corners_ref, ids_ref, rejected_ref = detector.detectMarkers(frame_ref)
            corners_ref, ids_ref, _, _ = detector.refineDetectedMarkers(frame_ref, ref_board, corners_ref, ids_ref, rejected_ref, cameraMatrix, distCoeffs)
            
            if ids_ref is not None:
                corners_ref = np.array(corners_ref) + (xmin_ref, ymin_ref)
                corners.append(corners_ref)
                ids.append(ids_ref)
            
            if ids_target is None and ids_ref is None:
                ids = None
            else:   
                ids = np.vstack(ids)
                corners = np.vstack(corners, dtype=np.float32)
        else:
            corners, ids, _ = detector.detectMarkers(frame)
            
        
        if STEREO:
            childConn.send((corners, ids))
        
        else:
            # pose, covariance = poseEstimator.estimate_pose_board(ref_board, target_board, corners, ids, cameraMatrix)
            # pose, covariance = poseEstimator.estimate_pose_board(ref_board, target_board, corners, ids, newCameraMatrix)
            pose, covariance = poseEstimator.estimate_pose_board(ref_board, target_board, corners, ids, cameraMatrix, distCoeffs)
            childConn.send((pose, covariance))

        if CROP and ids is not None and np.sum(ids<11) > 0 and np.sum(ids>=11) > 0:
            found = True
            corners = np.array(corners)
            xmin_target, ymin_target = np.max([[0,0],np.min(corners[ids<11].reshape(-1,2),axis=0)-CROP_PADDING],axis=0).astype(np.int32)
            xmax_target, ymax_target = np.min([[frame_width,frame_height],np.max(corners[ids<11].reshape(-1,2),axis=0)+CROP_PADDING],axis=0).astype(np.int32)
            xmin_ref, ymin_ref = np.max([[0,0],np.min(corners[ids>=11].reshape(-1,2),axis=0)-CROP_PADDING],axis=0).astype(np.int32)
            xmax_ref, ymax_ref = np.min([[frame_width,frame_height],np.max(corners[ids>=11].reshape(-1,2),axis=0)+CROP_PADDING],axis=0).astype(np.int32)
            
            if DISPLAY:
                frame = cv2.rectangle(frame, (xmin_target, ymin_target), (xmax_target, ymax_target),(0,255,0),2)
                frame = cv2.rectangle(frame, (xmin_ref, ymin_ref), (xmax_ref, ymax_ref),(0,255,0),2)
        elif CROP:
            found = False
            
        overlayImg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if DISPLAY: 
            cv2.imshow(f'Camera {cam}', overlayImg)
            
            if cv2.pollKey() == ord('q'):
                stopEvent.set()

    cap.release()
    childConn.close()
    cv2.destroyWindow(f'Camera {cam}')

def runPlot3D(childConn):
    interval = 1000/RATE
    fig_plot3d = plt.figure()
    # ax_target = fig_plot3d.add_subplot(1,2,1, projection='3d')
    # ax_ref = fig_plot3d.add_subplot(1,2,2, projection='3d')
    ax_target = fig_plot3d.add_subplot(projection='3d')
    # ax_target.set_xlim3d(-100,100)
    # ax_target.set_ylim3d(-100,100)
    # ax_target.set_zlim3d(180, 320)
    # points_ref = np.zeros((3,))
    points_target = np.zeros((3,))
    scatter_target = ax_target.scatter(points_target[0], points_target[1], points_target[2])
    # scatter_ref = ax_ref.scatter(points_ref[0], points_ref[1], points_ref[2])
    
    def animate3d(i):
        if childConn.poll():
            points_target, objPoints_target= childConn.recv()
            ax_target.clear()
            # ax_ref.clear()
            # ax_target.set(xlim=(-40,40), ylim=(-40,40), zlim=(215,285))
            ax_target.scatter(objPoints_target[0], objPoints_target[1], objPoints_target[2])
            ax_target.scatter(points_target[0], points_target[1], points_target[2])
            # ax_ref.scatter(points_ref[0], points_ref[1], points_ref[2])
    
    animplot = FuncAnimation(fig_plot3d, animate3d, interval=interval, cache_frame_data=False)
    plt.show()

def update_kalman(kalman: KalmanFilterCV, poses: list, covars: list):
    final_pose = kalman.predict().reshape((12,1))[0:6]
    # poses = [pose_1,pose_2,pose_3,pose_4,pose_5]
    # poses = []
    # covars = [covar_1,covar_2,covar_3,covar_4,covar_5]
    # covars = []
    kalman_measurement = np.array([])
    covariance_matrix = np.array([])
    num_cameras = 0
    for i in range(len(poses)):
        if poses[i] is not None:
            num_cameras += 1
            if len(kalman_measurement) == 0:
                kalman_measurement = poses[i]
                covariance_matrix = covars[i]        
            else:
                kalman_measurement = np.vstack((kalman_measurement,poses[i]))
                # Set size properly
                # [[1, 0],    [[1, 0, 0],
                #  [0, 1]] ->  [0, 1, 0]
                #              [0, 0, 1]]
                zero_block = np.zeros((covariance_matrix.shape[1],covars[i].shape[0]))
                covariance_matrix = np.block([[covariance_matrix,zero_block],
                                            [zero_block.T, covars[i]]])

    if num_cameras > 0:
        kalman.set_measurement(y_k=kalman_measurement) # 30x1 matrix; C matrix has to be 30x12 because also keeping track of x and x_dot
        kalman.set_measurement_matrices(num_measurements=num_cameras, new_R=covariance_matrix)
        # If can correct, return corrected position
        final_pose = kalman.correct()

    return kalman, final_pose

def reject_outliers(poses:list, m=2):
    poses = np.array(poses)
    distance_to_median = np.abs(poses - np.median(poses, axis=0))
    mdev = np.median(distance_to_median, axis=0)
    # Replace all zeros with 1
    mdev[mdev==0] = 1

    scaled_dist = distance_to_median / mdev 
    poses_no_outlier = []
    for i in range(len(poses)):
        if (np.mean(scaled_dist[i]) < m):
            poses_no_outlier.append(poses[i])
    
    return poses_no_outlier

def twoAngleMean(theta1, theta2):
    if abs(theta1-theta2) > 180:
        newtheta = ((theta1+theta2)/2 + 360) % 360 - 180
    else: 
        newtheta = (theta1+theta2)/2
    return newtheta

def anglesMean(thetas):
    if np.max(thetas) - np.min(thetas) > 180:
        avg = thetas[0]
        N = 1
        for theta in thetas[1:]:
            avg = twoAngleMean(2*avg*N/(N+1), 2*theta/(N+1))
            N += 1
        return avg
    else:
        return np.mean(thetas)
    
def rigidTransform(A, B, repeats=None):
    # Returns the R and t which transform the points in A towards those in B while minimizing least squares error
    # A and B must be of shape 3xN
    
    if repeats is not None:
        A = np.repeat(A, repeats, axis=1)
        B = np.repeat(B, repeats, axis=1)
    
    mean_A = np.mean(A, axis=1, keepdims=True)
    mean_B = np.mean(B, axis=1, keepdims=True)
    
    H = (A - mean_A) @ (B - mean_B).T
    U, S, Vh = np.linalg.svd(H)
    
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2,:] *= -1
        R = Vh.T @ U.T
    t = -R @ mean_A + mean_B
    
    return R, t

def triangulate(projMats, imgPoints):
    A = np.zeros((len(imgPoints)*2, 4), dtype=np.float32)
    A[::2] = imgPoints[:,[1]] * projMats[:,2] - projMats[:,1]
    A[1::2] = projMats[:,0] - imgPoints[:,[0]] * projMats[:,2]
    
    U, S, Vh = np.linalg.svd(A.T@A)
    worldPoint = Vh[3,:3]/Vh[3,3]
    
    return worldPoint

def doubleMAD(data):  # returns boolean array where inliers are True, based on doubleMAD outlier rejection
    median = hdquantiles(data, prob=0.5)[0]
    # mad_lower = 1.4826*hdquantiles(np.abs(data[data<=median]-median), prob=0.5)[0]
    mad_upper = 1.4826*hdquantiles(np.abs(data[data>=median]-median), prob=0.5)[0]
    # lower = median - 3*mad_lower
    upper = median + 3*mad_upper
    return data <= upper

def plot3D(pts, fig=None, ax=None):
    # pts should be 3xN
    if fig is None: fig = plt.figure()
    if ax is None: ax = fig.add_subplot(projection='3d')
    ax.scatter(pts[0], pts[1], pts[2])
    return fig, ax

def ros_publish(final_pose:np.ndarray, pose_msg):

    if final_pose is not None:
        # Translate for origin of marker object in gazebo
        pose_msg.pose.position.x = (final_pose[0]/1000) + -0.023
        pose_msg.pose.position.y = (final_pose[1]/1000) + -0.051
        pose_msg.pose.position.z = (final_pose[2]/1000) + 0
        euler = final_pose[3:].ravel()
        quat = R.from_euler(seq='ZYX',angles=euler,degrees=True).as_quat()
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
    
    return pose_msg

if __name__ == "__main__":
    if ROS: 
        rospy.init_node('pose_estimation', anonymous=True,log_level=rospy.INFO)
        # publisher = rospy.Publisher('kalman_filter/pose_estimate',Pose, queue_size=1)
        # pose_msg = Pose()
        
        publisher = rospy.Publisher('/gazebo/set_model_state',ModelState, queue_size=15,tcp_nodelay=True)
        # publisher = rospy.Publisher('/gazebo/set_model_state',ModelState, queue_size=1, tcp_nodelay=True)
        pose_msg = ModelState()
        pose_msg.model_name = 'surgical_pointer'
        rate = rospy.Rate(RATE)
        
    if IGT:
        server = pyigtl.OpenIGTLinkServer(port=IGT_Port)
        print("IGTL server started!")

    if KALMAN: kalman_filter = KalmanFilterCV(RATE)
    # kalman_filter.initiate_state(x0=np.zeros((6,1)))
    processes = []
    parentConns = []
    childConns = []
    stopEvent = mp.Event()
    barrier = mp.Barrier(len(cams)+1)
    
    # Importing camera matrices and distortion coefficients from calibFile
    calibData = pd.read_json(calibFile_camera, orient='records')
    for i, id in enumerate(calibData.loc[:,'id']): 
        calibData.at[i,'id'] = tuple(id) if type(id) == list else id
    calibData = calibData.set_index('id')
    calibData = calibData.applymap(np.array)
    calibData = calibData.replace(np.nan, None)
    cameraMatrices = np.array(calibData.loc[cams, 'cameraMatrix'])
    # cameraMatrices = np.zeros((N_cams,3,3), dtype=np.float32)
    distCoeffs_all = np.array(calibData.loc[cams, 'distCoeffs'])
    R_toWorld = np.zeros((N_cams,3,3), dtype=np.float32)  # R_toWorld and T_toWorld are used to transform points from camera reference frames to world
    T_toWorld = np.zeros((N_cams,3,1), dtype=np.float32)
    projMats = np.zeros((N_cams, 3, 4), dtype=np.float32)  # projective matrices that transform 3D points in world frame to image points in each camera
    for i, cam in enumerate(cams):
        # cameraMatrices[i] = cv2.getOptimalNewCameraMatrix(calibData.at[cam, "cameraMatrix"], calibData.at[cam, "distCoeffs"], imageSize, alpha=1, newImgSize=imageSize)[0]
        R_fromWorld, T_fromWorld = calibData.at[cam, 'R'], calibData.at[cam, 'T']
        R_toWorld[i] = R_fromWorld.T
        T_toWorld[i] = -R_fromWorld.T @ T_fromWorld
        projMats[i] = cameraMatrices[i] @ np.hstack([R_fromWorld,T_fromWorld])
    cams_indices = np.arange(len(cams))
    camIndices_dict = dict((cam, camIndex) for camIndex, cam in enumerate(cams_all))
    camPairs_indices = list(itertools.combinations(list(range(N_cams)), 2))
    
    # Initialize lists to store data if plotting is True
    if PLOTTING:
        start_time = time.time()
        times = []
        pose_estimates = []

    for cam in cams:
        cameraMatrix = calibData.at[cam, "cameraMatrix"]
        distCoeffs = calibData.at[cam, "distCoeffs"]
        parentConn, childConn = mp.Pipe(False)  # True for two-way communication, False for one-way
        process = mp.Process(target=runCam, args=(cam, cameraMatrix, distCoeffs, childConn, stopEvent, barrier))
        process.start()
        
        processes.append(process)
        parentConns.append(parentConn)
        childConns.append(childConn)
        
    if PLOTTING_3D:
        parentConn_plot3d, childConn_plot3d = mp.Pipe(True)
        process_plot3d = mp.Process(target=runPlot3D, args=(childConn_plot3d,))
        process_plot3d.start()
    
    # lastPublish = time.time()
    final_pose = np.zeros((6,1))

    # List to store timestamps for each camera
    timestamps = [0] * len(cams)
    
    frameTime = 1/RATE
    frame = 0
    lastFrameTime = time.time()
    barrier.wait()
    
    if KALMAN and not ADAPTIVE:
        if STEREO:
            Q_matrix = np.loadtxt(QFile_stereo, delimiter=',')  # Load Q matrix from file
            # Q_diag = np.diag(Q_matrix)
            # init_covars = Q_diag.reshape(-1, 1) # Reshape into a 6x1 column vector
            init_covars = [Q_matrix]
        else:
            R_matrix = np.loadtxt(RFile_mono, delimiter=',')  # Load the R matrix from the .txt file
            init_covars = []
            for cam in cams:
                start_idx = camIndices_dict[cam] * 6  # Starting index for this camera (0, 6, 12, 18, 24)
                end_idx = start_idx + 6  # Ending index for this camera (6, 12, 18, 24, 30)
                init_covar = np.diag(R_matrix[start_idx:end_idx, start_idx:end_idx])  # Extract the appropriate diagonal section
                init_covar = np.diag(init_covar)
                init_covars.append(init_covar)

    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        if (currTime:=time.time()) - lastFrameTime < frameTime: continue
        dt = currTime-lastFrameTime
        if KALMAN: kalman_filter.set_dt(dt)
        print(f"Currently running at {1/dt:.2f} FPS", end='\r')
        lastFrameTime = currTime
        frame += 1
        
        if KALMAN: kalman_filter.set_dt(dt)
        
        if True not in (process.is_alive() for process in processes): break
        if ROS and rospy.is_shutdown(): 
            cv2.destroyAllWindows()
            break
        
        poses = []
        poses_full = [None]*N_cams
        zero_covars = []
        
        if STEREO:
            allCorners = np.full((N_tagIds, N_cams, 4, 2), np.nan, dtype=np.float32)  # 4 corners per tag, x & y at each corner
            for camIndex, cam in enumerate(cams):
                corners, ids = parentConns[camIndex].recv()
                if ids is not None:
                    corners_array = np.array(corners, dtype=np.float32).reshape(-1,2)
                    corners_undist = cv2.undistortImagePoints(corners_array, cameraMatrices[camIndex], distCoeffs_all[camIndex])
                    allCorners[ids,camIndex,:,:] = corners_undist.reshape(-1,1,4,2)
            barrier.wait()
            
            foundTags = np.any(~np.isnan(allCorners), axis=(2,3))  # Boolean array of N_tags x N_cams
            if np.max(np.sum(foundTags[:11], axis=1)) > 1 and np.max(np.sum(foundTags[11:], axis=1)) > 1:
                # reprojectTags = [np.where(foundTags[:,camIndex])[0] for camIndex in cams_indices]  # reproject all tags visible to a camera
                reprojectTags = [np.where(np.logical_and(foundTags[:,camIndex], foundTags.sum(axis=1)==1))[0] for camIndex in cams_indices]
                triangulateTags = [np.where(np.all(foundTags[:,[cam1,cam2]], axis=1))[0].tolist() for cam1, cam2 in camPairs_indices]  # Tag pairs for use with cv2.triangulatePoints
                triangulateCams = [np.where(foundTags[tag])[0].tolist() for tag in range(N_tagIds)]  # Cam lists per tag for use with multi-cam triangulate function
                
                objPoints_all = []
                worldPoints_all = []
                tags_all = []
                
                if STEREO_REPROJECT:
                    for tags_reproject, camIndex in zip(reprojectTags, cams_indices):
                        if len(tags_reproject) == 0: continue
                        tags_found = np.where(foundTags[:,camIndex])[0]  # all tags visible to this camera
                        imgPoints = allCorners[tags_found, camIndex]
                        objPoints_target, imgPoints_target = target_board.matchImagePoints(imgPoints,tags_found)
                        objPoints_ref, imgPoints_ref = ref_board.matchImagePoints(imgPoints,tags_found)
                        
                        ret_target, rvec_target, tvec_target = cv2.solvePnP(objPoints_target, imgPoints_target, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_ITERATIVE)
                        ret_ref, rvec_ref, tvec_ref = cv2.solvePnP(objPoints_ref, imgPoints_ref, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_ITERATIVE)

                        cv2.solvePnPRefineLM(objPoints_target, imgPoints_target, cameraMatrices[camIndex], None, rvec_target, tvec_target, criteria_refineLM)
                        cv2.solvePnPRefineLM(objPoints_ref,imgPoints_ref, cameraMatrices[camIndex], None, rvec_ref, tvec_ref, criteria_refineLM)
                        
                        objPoints_target_reproject = tagPoints[tags_reproject[tags_reproject<11]]
                        objPoints_ref_reproject = tagPoints[tags_reproject[tags_reproject>=11]]
                        objPoints_reproject = np.vstack([objPoints_target_reproject, objPoints_ref_reproject])
                        
                        camPoints_target = cv2.Rodrigues(rvec_target)[0] @ objPoints_target_reproject.reshape((-1,3)).T + tvec_target  # camPoints and worldPoints are column vectors
                        camPoints_ref = cv2.Rodrigues(rvec_ref)[0] @ objPoints_ref_reproject.reshape((-1,3)).T + tvec_ref
                        
                        worldPoints = R_toWorld[camIndex] @ np.hstack([camPoints_target, camPoints_ref]) + T_toWorld[camIndex]
                        
                        objPoints_all.append(objPoints_reproject.reshape((-1,4,3)))
                        worldPoints_all.append(worldPoints.T.reshape((-1,4,3)))
                        tags_all += tags_reproject.tolist()

                if STEREO_TRIANGULATE_PAIRS:
                    # Triangulation of pairs
                    for tags, (cam1, cam2) in zip(triangulateTags, camPairs_indices):
                        if len(tags) == 0: continue
                        projMat1, projMat2 = projMats[[cam1,cam2]]
                        imgPoints1 = allCorners[tags, cam1].reshape((-1,2)).T
                        imgPoints2 = allCorners[tags, cam2].reshape((-1,2)).T
                        objPoints = tagPoints[tags].reshape((-1,4,3))
                        
                        worldPoints = cv2.triangulatePoints(projMat1, projMat2, imgPoints1, imgPoints2)
                        worldPoints = cv2.convertPointsFromHomogeneous(worldPoints.T).reshape((-1,4,3))
                        
                        objPoints_all.append(objPoints)
                        worldPoints_all.append(worldPoints)
                        tags_all += tags
                
                if STEREO_TRIANGULATE_ALL:
                    # Triangulation across every view
                    for tag, camIndices in zip(np.arange(N_tagIds), triangulateCams):
                        if len(camIndices) < 2: continue
                        objPoints = tagPoints[tag].reshape((1,4,3))
                        worldPoints = np.zeros((1,4,3))
                        # if REJECT_OUTLIER_TAGS: reprojErrors = np.zeros((4,len(camIndices)))
                        
                        for corner in range(4):
                            imgPoints = allCorners[tag, camIndices, corner]
                            worldPoints[0,corner] = triangulate(projMats[camIndices], imgPoints)
                            
                            if REJECT_OUTLIER_TAGS:
                                imgPoints_reproj = projMats[camIndices] @ np.vstack([worldPoints[0,corner].reshape(3,1),1])
                                imgPoints_reproj = (imgPoints_reproj.reshape(-1,3) / imgPoints_reproj[:,-1])[:,:2]
                                # reprojErrors[corner] = np.linalg.norm(imgPoints_reproj - imgPoints, axis=1)
                                reprojErrors = np.linalg.norm(imgPoints_reproj - imgPoints, axis=1)
                                if np.max(reprojErrors) > REJECT_THRESH: break
                        else:
                        
                            # if REJECT_OUTLIER_TAGS and np.max(reprojErrors) > REJECT_THRESH:
                            #     continue
                            
                            objPoints_all.append(objPoints)
                            worldPoints_all.append(worldPoints)
                            tags_all.append(tag)
                    
                    
                tags_all = np.array(tags_all)
                objPoints_all = np.vstack(objPoints_all)
                objPoints_target = objPoints_all[tags_all<11].reshape((-1,3)).T
                objPoints_ref = objPoints_all[tags_all>=11].reshape((-1,3)).T
                
                worldPoints_all = np.vstack(worldPoints_all)
                worldPoints_target = worldPoints_all[tags_all<11].reshape((-1,3)).T
                worldPoints_ref = worldPoints_all[tags_all>=11].reshape((-1,3)).T
                
                if STEREO_TRIANGULATE_ALL:
                    # repeats_target = np.repeat(np.sum(foundTags, axis=1)[tags_all[tags_all<11]]-1, 4)
                    # repeats_ref = np.repeat(np.sum(foundTags, axis=1)[tags_all[tags_all>=11]]-1, 4)
                    
                    repeats_target = np.array([np.sum(np.arange(1,n)) for n in np.sum(foundTags, axis=1)[tags_all[tags_all<11]]])
                    repeats_target = np.repeat(repeats_target, 4)
                    repeats_ref = np.array([np.sum(np.arange(1,n)) for n in np.sum(foundTags, axis=1)[tags_all[tags_all>=11]]])
                    repeats_ref = np.repeat(repeats_ref, 4)
                    # repeats_target, repeats_ref = None, None
                else:
                    repeats_target, repeats_ref = None, None
                
                R_target, t_target = rigidTransform(objPoints_target, worldPoints_target, repeats_target)
                R_ref, t_ref = rigidTransform(objPoints_ref, worldPoints_ref, repeats_ref)
                
                # if REJECT_OUTLIER_TAGS:
                #     errs_target = np.linalg.norm(R_target @ objPoints_target + t_target - worldPoints_target, axis=0)
                #     errs_ref = np.linalg.norm(R_ref @ objPoints_ref + t_ref - worldPoints_ref, axis=0)
                #     filteredCorners_target = doubleMAD(errs_target)
                #     filteredCorners_ref = doubleMAD(errs_ref)
                #     if np.sum(~filteredCorners_target):
                #         objPoints_target, worldPoints_target, repeats_target = objPoints_target[:,filteredCorners_target], worldPoints_target[:,filteredCorners_target], repeats_target[filteredCorners_target]
                #         R_target, t_target = rigidTransform(objPoints_target, worldPoints_target, repeats_target)
                #     if np.sum(~filteredCorners_ref):
                #         objPoints_ref, worldPoints_ref, repeats_ref = objPoints_ref[:,filteredCorners_ref], worldPoints_ref[:,filteredCorners_ref], repeats_ref[filteredCorners_ref]
                #         R_ref, t_ref = rigidTransform(objPoints_ref, worldPoints_ref, repeats_ref)
                
                rel_trans = R_ref.T @ (t_target - t_ref)
                rel_rot_matrix = R_target.T @ R_ref
                rel_rot_ypr = R.from_matrix(rel_rot_matrix).as_euler('ZYX',degrees=True).reshape((3,1))
                pose = np.vstack((rel_trans, rel_rot_ypr))
                
                if not np.isnan(pose).any():
                    poses.append(pose)
                    poses_full[0] = pose
                    if KALMAN and not ADAPTIVE: zero_covars = init_covars
                    
                    if PLOTTING_3D:
                        objToWorldPoints_target = R_target @ tagPoints[:11].reshape((-1,3)).T + t_target
                        parentConn_plot3d.send((worldPoints_target, objToWorldPoints_target))
                        # objToWorldPoints_ref = R_ref @ tagPoints[11:].reshape((-1,3)).T + t_ref
                        # parentConn_plot3d.send((worldPoints_ref, objToWorldPoints_ref))
                        # worldToObjPoints_target = R_target.T @ worldPoints_target - R_target.T @ t_target
                        # parentConn_plot3d.send((worldToObjPoints_target, tagPoints[:11].reshape((-1,3)).T))
            
        else:
            for i, cam in enumerate(cams):
                pose, covar = parentConns[i].recv()
                if pose is not None:
                    poses.append(pose)
                    poses_full[i] = pose
                    if KALMAN and not ADAPTIVE: zero_covars.append(init_covars[i])
                else: poses_full.append(pose)
            barrier.wait()
        
        if ADAPTIVE:
            covars = kalman_filter.compute_measurement_noise(poses=poses_full)
            # Find indices where elements in poses_full are None
            none_indices = [i for i, pose in enumerate(poses_full) if pose is None]
            # Remove elements from covars at the same indices
            covars = [covar for i, covar in enumerate(covars) if i not in none_indices]
            kalman_filter.set_process_noise(covars)

        if len(poses) > 0:
            if KALMAN:
                if kalman_filter.has_been_initiated():
                    if REJECT_OUTLIER_CAMS:
                        poses_no_outliers = reject_outliers(poses=poses, m = 2)
                        kalman_filter, final_pose = update_kalman(kalman_filter, poses=poses_no_outliers, covars=covars)
                    else:
                        if ADAPTIVE:
                            kalman_filter, final_pose = update_kalman(kalman_filter, poses=poses, covars=covars)
                            final_pose = final_pose[0:6]
                        else:
                            kalman_filter, final_pose = update_kalman(kalman_filter, poses=poses, covars=zero_covars)
                            final_pose = final_pose[0:6]
                        # poses = np.array(poses)
                        # for i in range(3,6):
                        #     final_pose[i] = anglesMean(poses[:,i])
                else:
                    kalman_filter.initiate_state(x0=np.median(poses,axis=0))
                    final_pose = kalman_filter.predict().reshape((12,1))[0:6]
            else:
                poses = np.array(poses)
                final_pose[:3] = np.mean(poses[:,:3], axis=0)
                for i in range(3,6):
                    final_pose[i] = anglesMean(poses[:,i])
        
        if PLOTTING:
            curr_time = time.time() - start_time
            if curr_time >= 7 and curr_time <= 19:
                times.append(curr_time)
                pose_estimates.append(final_pose.flatten().tolist())
            else: continue

        if ROS:
            if len(poses)>0:
                pose_msg = ros_publish(final_pose, pose_msg)
                publisher.publish(pose_msg)
                
            rate.sleep()

        if IGT:
            transform = np.eye(4)
            transform[:3,:3] = R.from_euler(seq='ZYX',angles=final_pose[3:].ravel(),degrees=True).as_matrix().T # Transpose to convert internal rotation vector format to external ibis format
            transform[:3,3] = final_pose[:3].flatten()
            position_message = pyigtl.TransformMessage(transform, device_name="PointerDevice")
            server.send_message(position_message, wait=False)
        
    if IGT: 
        server.stop()
        print('Server port closed succesfully!')
    
    if PLOTTING:
        # Ensure lists are numpy arrays before plotting
        times = np.array(times)
        pose_estimates = np.array(pose_estimates)

        # Define figure and axes
        fig, axs = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
        pose_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']

        for i in range(3):  # XYZ on the left column
            # Calculate mean and standard deviations
            mean_pose = np.mean(pose_estimates[:, i])
            std_dev_pose = np.std(pose_estimates[:, i])
            one_sigma = std_dev_pose
            two_sigma = 2 * std_dev_pose

            # Plot
            line1, = axs[i, 0].plot(times, pose_estimates[:, i], label=f'{pose_labels[i]} Estimate')
            mean_line = axs[i, 0].axhline(mean_pose, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_pose:.2f}')
            one_sigma_line1 = axs[i, 0].axhline(mean_pose + one_sigma, color='blue', linestyle='--', linewidth=1, label=f'1σ: ±{one_sigma:.2f}')
            one_sigma_line2 = axs[i, 0].axhline(mean_pose - one_sigma, color='blue', linestyle='--', linewidth=1)
            two_sigma_line1 = axs[i, 0].axhline(mean_pose + two_sigma, color='green', linestyle='--', linewidth=1, label=f'2σ: ±{two_sigma:.2f}')
            two_sigma_line2 = axs[i, 0].axhline(mean_pose - two_sigma, color='green', linestyle='--', linewidth=1)

            # Add legend with mean and standard deviation values
            handles = [line1, mean_line, one_sigma_line1, two_sigma_line1]
            axs[i, 0].legend(handles=handles, loc='upper right')

            axs[i, 0].set_xlabel('Time (s)')
            axs[i, 0].set_ylabel('Position (mm)')

        for i in range(3, 6):  # Roll, Pitch, Yaw on the right column
            # Calculate mean and standard deviations
            mean_pose = np.mean(pose_estimates[:, i])
            std_dev_pose = np.std(pose_estimates[:, i])
            one_sigma = std_dev_pose
            two_sigma = 2 * std_dev_pose

            # Plot
            line1, = axs[i-3, 1].plot(times, pose_estimates[:, i], label=f'{pose_labels[i]} Estimate')
            mean_line = axs[i-3, 1].axhline(mean_pose, color='red', linestyle='--', linewidth=1, label=f'Mean: {mean_pose:.2f}')
            one_sigma_line1 = axs[i-3, 1].axhline(mean_pose + one_sigma, color='blue', linestyle='--', linewidth=1, label=f'1σ: ±{one_sigma:.2f}')
            one_sigma_line2 = axs[i-3, 1].axhline(mean_pose - one_sigma, color='blue', linestyle='--', linewidth=1)
            two_sigma_line1 = axs[i-3, 1].axhline(mean_pose + two_sigma, color='green', linestyle='--', linewidth=1, label=f'2σ: ±{two_sigma:.2f}')
            two_sigma_line2 = axs[i-3, 1].axhline(mean_pose - two_sigma, color='green', linestyle='--', linewidth=1)

            # Add legend with mean and standard deviation values
            handles = [line1, mean_line, one_sigma_line1, two_sigma_line1]
            axs[i-3, 1].legend(handles=handles, loc='upper right')

            axs[i-3, 1].set_xlabel('Time (s)')
            axs[i-3, 1].set_ylabel('Angle (degrees)')

        plt.show()


        # Plotting true v. estimated pose

        fixed_points = np.array([
            # [77.24, -30.5544, 21.0519],
            [39.5306, -9.2923, 85.0889],
            [-14.3043, -72.0667, 72.605],
            [19.3012, -88.9711, 3.9566],
            [31.4759, -11.4459, -66.5619],
            # [77.5623, -19.2732, -3.6433],
            # [40.9331, -80.6808, 0.0667],
            # [5.6701, -80.7094, -36.2677]
        ])

        # moving_points = np.array([
        #     [233.35909978, 29.30010574, 32.66220421],
        #     [161.49023128, 43.00181951, 11.0627511],
        #     [138.72047237, -3.48904037, 76.51294322],
        #     [210.30778731, -21.16662992, 89.74480555],
        # ])

        no_kalman_points = np.array([
            # [157.008, 59.3237, -358.971],
            [135.524, 83.532, 14.563],
            [118.43, 30.5293, 72.0615],
            [197.714, 29.4515, 98.5277],
            [264.529, 4.4746, 5.85285],
            # [230.63, 77.5104, 28.2496],
            # [213.605, 45.7472, 87.9976],
            # [225.214, -3.51287, 87.2148]
        ])

        non_adaptive_points = np.array([
            # [157.008, 59.3237, -358.971],
            [135.882, 84.0918, 16.9069],
            [118.794, 31.137, 73.4534],
            [198.332, 29.5957, 99.6988],
            [264.491, 4.61425, 6.29104],
            # [230.536, 77.9316, 28.1567],
            # [213.825, 45.7351, 87.979],
            # [226.354, -3.0305, 88.5085]
        ])

        adaptive_points = np.array([
            # [157.008, 59.3237, -358.971],
            [136.869, 81.019, 16.1089],
            [118.886, 30.1037, 72.4736],
            [195.448, 26.1335, 95.1602],
            [264.27, 1.41152, 9.84017],
            # [231.297, 70.7279, 16.425],
            # [209.738, 42.1161, 85.0089],
            # [223.178, -3.7345, 83.751]
        ])

        R_reg, T_reg, _ = register(fixed_points, adaptive_points)
        transformed_poses = apply_transform(pose_estimates[:, :3], R_reg, T_reg)

        # Select one of the fixed points as the true value
        # true_value = fixed_points[5]
        # true_value = [77.5623, -19.2732, -3.6433]
        # true_value = [40.9331, -80.6808, 0.0667]
        true_value = [5.6701, -80.7094, -36.2677]

        # Calculate errors between transformed poses and true value
        errors = transformed_poses - true_value

        # Define figure and axes for error plots
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
        pose_labels = ['X', 'Y', 'Z']

        for i in range(3):  # Loop over X, Y, Z
            # Calculate mean error and standard deviations
            mean_error = np.mean(errors[:, i])
            std_dev_error = np.std(errors[:, i])
            one_sigma_error = std_dev_error
            two_sigma_error = 2 * std_dev_error

            # Plot
            line1, = axs[i].plot(times, transformed_poses[:, i], label=f'{pose_labels[i]} Estimate')
            truth_line = axs[i].axhline(true_value[i], color='purple', linestyle='-', linewidth=1, label=f'Truth: {true_value[i]:.2f}')
            mean_error_line = axs[i].axhline(true_value[i] + mean_error, color='red', linestyle='--', linewidth=1, label=f'Mean Error: {mean_error:.2f}')
            one_sigma_error_line1 = axs[i].axhline(true_value[i] + mean_error + one_sigma_error, color='blue', linestyle='--', linewidth=1, label=f'1σ: ±{one_sigma_error:.2f}')
            one_sigma_error_line2 = axs[i].axhline(true_value[i] + mean_error - one_sigma_error, color='blue', linestyle='--', linewidth=1)
            two_sigma_error_line1 = axs[i].axhline(true_value[i] + mean_error + two_sigma_error, color='green', linestyle='--', linewidth=1, label=f'2σ: ±{two_sigma_error:.2f}')
            two_sigma_error_line2 = axs[i].axhline(true_value[i] + mean_error - two_sigma_error, color='green', linestyle='--', linewidth=1)

            # Add legend with mean error and standard deviation values
            handles = [line1, truth_line, mean_error_line, one_sigma_error_line1, two_sigma_error_line1]
            axs[i].legend(handles=handles, loc='upper right')

            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel(f'{pose_labels[i]} (mm)')

            # Calculate RMS of the entire pose error
            combined_rms_error = np.sqrt(np.mean(np.sum(errors**2, axis=1)))

            # Add combined RMS errors text at the bottom of the plot
            fig.text(0.5, 0.01, f'XYZ RMSE: {combined_rms_error:.4f}', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        plt.show()