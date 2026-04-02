# %%
import numpy as np
import cv2
from scipy.optimize import least_squares
import pandas as pd
import time
import multiprocessing as mp
import itertools
from pynput import keyboard
from scipy.stats.mstats import hdquantiles

LIVE = False
DISPLAY = False
INTRINSIC = True  # Set to True to calibrate intrinsic params (camera matrix and distortion coefficients for each camera)
EXTRINSIC = True  # Set to True to calibrate extrinsic params (translation vector and rotation matrix for each camera pair)
MINPOS = True  # Bundle adjustment minimizes positional error instead of reprojection error
TAGBASED = False  # Uses corners of aruco tag grid for calibration, otherwise the checkerboard corners of a charuco board
BUNDLEINT = False
SAVE_IMAGES = False  # Saves live calibration images to new folder in calib_images/old/ if True
cams = [1,2,3,4,5]  # List of cameras to calibrate, all of which must be included in camList

# initCalib_path = 'calib_files/camera/2025-06-13_11_02_10.json'  # Calibration file to use for initial values; can only be set to None if INTRINSIC is True
initCalib_path = None
videoPath_int = 'calib_videos/intrinsic_T33'  # folder to grab video from if VIDEO is true but LIVE is false
videoPath_ext = 'calib_videos/extrinsic_T53_1'  # folder to grab video from if VIDEO is true but LIVE is false
idList = [1, 2, 3, 4, 5]  # list of all camera IDs that will be included in the calibration file (not necessarily being calibrated now)

captureRate = 100  # fps at which frames are synchronously grabbed from all cameras for calibration
frameSkip_int = 4  # skip calibration of every N frames captured
frameSkip_ext = 2  # skip calibration of every N frames captured
outlierCoeff = 3

minPoints = 35
if TAGBASED: minTags = 24
imageSize = 1456, 1088

# Charuco params
patternSize = (6, 8)  # number of squares in X, Y
squareLength = 32.004  # in mm
markerLength = 17.992  # in mm
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
if TAGBASED: 
    N_tags = int(patternSize[0]*patternSize[1]/2)
    N_corners = N_tags * 4
else: N_corners = (patternSize[0]-1) * (patternSize[1]-1)  # Total number of charuco corners, used to verify full board visibility

camPairs = list(itertools.combinations(cams, 2))

if initCalib_path:
    calibData = pd.read_json(initCalib_path, orient='records')
    for i, id in enumerate(calibData.loc[:,'id']): 
        calibData.at[i,'id'] = tuple(id) if type(id) == list else id
    calibData = calibData.set_index('id')
    calibData = calibData.applymap(np.array)
    calibData = calibData.replace(np.nan, None)
else:
    calibData = pd.DataFrame(index=idList, columns=["cameraMatrix", "distCoeffs", "R", "T"])
    calibData = calibData.replace(np.nan, None)

charucoBoard = cv2.aruco.CharucoBoard(patternSize, squareLength, markerLength, arucoDict)
# if TAGBASED: arucoBoard = cv2.aruco.Board(charucoBoard.getObjPoints(), arucoDict, charucoBoard.getIds())
detectorParams = cv2.aruco.DetectorParameters()
detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # Default CORNER_REFINE_NONE
detectorParams.cornerRefinementMaxIterations = 1000  # Default 30
detectorParams.cornerRefinementMinAccuracy = 0.001  # Default 0.1
detectorParams.adaptiveThreshWinSizeStep = 2  # Default 10
detectorParams.adaptiveThreshWinSizeMax = 15  # Default 23
detectorParams.adaptiveThreshConstant = 8  # Default 7

charucoDetectors = dict()
for cam in cams:
    charucoParams = cv2.aruco.CharucoParameters()
    charucoParams.tryRefineMarkers = False  # Default False
    charucoParams.cameraMatrix = calibData.at[cam, "cameraMatrix"]
    charucoParams.distCoeffs = calibData.at[cam, "distCoeffs"]
    charucoDetectors[cam] = cv2.aruco.CharucoDetector(charucoBoard, charucoParams, detectorParams)
    
criteria_refineLM = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 200, 1.19209e-07)

N_cams = len(cams)

def updateDetectors(calibData):
    charucoDetectors = dict()
    for cam in cams:
        charucoParams = cv2.aruco.CharucoParameters()
        charucoParams.tryRefineMarkers = False  # Default False
        charucoParams.cameraMatrix = calibData.at[cam, "cameraMatrix"]
        charucoParams.distCoeffs = calibData.at[cam, "distCoeffs"]
        charucoDetectors[cam] = cv2.aruco.CharucoDetector(charucoBoard, charucoParams, detectorParams)
    return charucoDetectors

def runCam(videoPath, cam, frameSkip, childConn, stopEvent, barrier):
    detector = charucoDetectors[cam]
    if LIVE: cap = cv2.VideoCapture(f"udpsrc address=192.168.3.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
    else: cap = cv2.VideoCapture(f'{videoPath}/{cam}.mp4')
    if not cap.isOpened():
        print(f"Cannot open camera {cam}.")
        return
    
    frameIndex = 0
    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        if cv2.pollKey() == ord('q'):
            cv2.destroyWindow(f'Camera {cam}')
            break
        
        barrier.wait()
        ret, frame = cap.read()  # ret is True if frame is read correctly
        
        if not ret:
            if not LIVE and not stopEvent.is_set():
                print("Prerecorded video finished.")
                stopEvent.set()
            else:
                print(f"Can't receive frame from camera {cam}.")
            childConn.send((None, None, None))
            continue
        
        message = [None] * 3
        if frameIndex % frameSkip == 0:
            charucoCorners, charucoIds, arucoMarkerCorners, arucoMarkerIds = detector.detectBoard(frame)
        
            if charucoCorners is not None:
                if DISPLAY: frame = cv2.aruco.drawDetectedMarkers(frame, arucoMarkerCorners, arucoMarkerIds)
                if TAGBASED and len(arucoMarkerIds) >= minTags and np.max(arucoMarkerIds)<N_tags: 
                    message = [frame, arucoMarkerCorners, arucoMarkerIds]
                elif not TAGBASED and len(charucoIds) >= minPoints: 
                    message = [frame, charucoCorners, charucoIds]

        childConn.send(message)
        
        if DISPLAY: cv2.imshow(f'Camera {cam}', frame)
        frameIndex += 1

    cap.release()
    childConn.close()
    cv2.destroyWindow(f'Camera {cam}')

def processVideos(videoPath, extrinsic):
    processes = dict()
    parentConns = dict()
    childConns = dict()
    barrier = mp.Barrier(len(cams)+1, timeout=10)
    stopEvent = mp.Event()
    listener = keyboard.Listener(on_press=onPress)
    listener.start()
    frameSkip = frameSkip_ext if extrinsic else frameSkip_int
    
    for cam in cams:
        parentConn, childConn = mp.Pipe(True)  # True for two-way communication, False for one-way
        process = mp.Process(target=runCam, args=(videoPath, cam, frameSkip, childConn, stopEvent, barrier))
        process.start()

        processes[cam] = process
        parentConns[cam] = parentConn
        childConns[cam] = childConn
    
    capTiming = 1/captureRate
    capIndex = 0  # current index of simultaneous captures where at least one camera found the board
    capCounts = dict([(cam, 0) for cam in cams])  # current number of successful captures per camera
    if extrinsic: capCounts.update([(camPair, 0) for camPair in camPairs])
    lastCapTime = -capTiming
    # if SAVE_IMAGES: 
    #     newImages_path = f'calib_images/{time.strftime("%Y-%m-%d %H_%M_%S")}'
    #     os.mkdir(newImages_path)
    #     for cam in cams: os.mkdir(f"{newImages_path}/{cam}")
    imgPoints_dict = dict([(cam, [None]*500) for cam in cams])  # Each simultaneous capture appends an inner list including numpy arrays of corners found by each camera, or None for any that didn't see it
    objPoints_dict = dict([(cam, [None]*500) for cam in cams])
    # ids_dict = dict([(cam, [None]*500) for cam in cams])
    # imageSizes = dict([(cam, None) for cam in cams])
    
    if TAGBASED: objPoints = np.array(charucoBoard.getObjPoints()).reshape(-1,1,3)
    
    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        if time.time() - lastCapTime >= capTiming:
            lastCapTime = time.time()
            barrier.wait()
            boardFound = False
            
            for cam in cams:
                frame, corners, ids = parentConns[cam].recv()
                if frame is not None:
                    if TAGBASED:
                        # objPoints, imgPoints = arucoBoard.matchImagePoints(corners, ids)
                        imgPoints = np.full((len(ids),4,2), np.nan, dtype=np.float32)
                        imgPoints[ids] = corners
                        objPoints_dict[cam][capIndex] = objPoints
                        imgPoints_dict[cam][capIndex] = imgPoints.reshape(-1,1,2)
                    else:
                        objPoints, imgPoints = charucoBoard.matchImagePoints(corners, ids)
                        objPoints_dict[cam][capIndex] = objPoints
                        imgPoints_dict[cam][capIndex] = imgPoints
                    # imageSizes[cam] = frame.shape[1::-1]
                    boardFound = True
                    capCounts[cam] += 1
                    # if SAVE_IMAGES: cv2.imwrite(f'{newImages_path}/{cam}/{capIndex}.jpg', frame)
            
            if boardFound: 
                if extrinsic:
                    for cam1, cam2 in camPairs:
                        if imgPoints_dict[cam1][capIndex] is not None and imgPoints_dict[cam2][capIndex] is not None: capCounts[cam1,cam2] += 1
                
                capIndex += 1
                if capIndex % 500 == 0: 
                    for cam in cams: 
                        imgPoints_dict[cam] += [None]*500
                        objPoints_dict[cam] += [None]*500
                        # ids_dict[cam] += [None]*500
                
                print(f'Captures: {capCounts}', end='\r')
    print(f'Captures: {capCounts}')            
    
    imgPoints_df = pd.DataFrame(imgPoints_dict, columns=idList)
    objPoints_df = pd.DataFrame(objPoints_dict, columns=idList)
    
    return imgPoints_df, objPoints_df

def intrinsicCalib(imgPoints_df, objPoints_df):
    intrinsicErrors = dict([(cam, None) for cam in cams])
    perViewErrors = dict([(cam, None) for cam in cams])
    for cam in cams:
        print(f'Starting intrinsic calibration for cam {cam}.')
        initCameraMatrix = calibData.at[cam, "cameraMatrix"]
        initDistCoeffs = calibData.at[cam, "distCoeffs"]
        imgPoints = list(imgPoints_df.loc[:,cam].dropna())
        objPoints = list(objPoints_df.loc[:,cam].dropna())
        
        calibFlags = cv2.CALIB_USE_INTRINSIC_GUESS if initCameraMatrix is not None else 0
        error, cameraMatrix, distCoeffs, _, _, _, _, perViewError = cv2.calibrateCameraExtended(objPoints, imgPoints, imageSize, initCameraMatrix, initDistCoeffs, flags=calibFlags)
        
        calibData.at[cam, "cameraMatrix"] = cameraMatrix
        calibData.at[cam, "distCoeffs"] = distCoeffs
        intrinsicErrors[cam] = error
        perViewErrors[cam] = perViewError
    return intrinsicErrors, perViewErrors

def onPress(key):
    global captureEvent, stopEvent
    try:
        if key.char == 'q':
            stopEvent.set()
            print("\nStopping calibration.")
    except AttributeError:
        
        pass
    
def triangulate(projMats, imgPoints):
    A = np.zeros((len(imgPoints)*2, 4), dtype=np.float32)
    A[::2] = imgPoints[:,[1]] * projMats[:,2] - projMats[:,1]
    A[1::2] = projMats[:,0] - imgPoints[:,[0]] * projMats[:,2]
    
    U, S, Vh = np.linalg.svd(A.T@A)
    worldPoint = Vh[3,:3]/Vh[3,3]
    
    return worldPoint

def rigidTransform(A, B):
    # Returns the R and t which transform the points in A towards those in B while minimizing least squares error
    # A and B must be of shape 3xN
    
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

def doubleMAD(data):  # returns boolean array where inliers are True, based on doubleMAD outlier rejection
    median = hdquantiles(data, prob=0.5)[0]
    # mad_lower = 1.4826*hdquantiles(np.abs(data[data<=median]-median), prob=0.5)[0]
    mad_upper = 1.4826*hdquantiles(np.abs(data[data>=median]-median), prob=0.5)[0]
    # lower = median - 3*mad_lower
    upper = median + outlierCoeff*mad_upper
    return data <= upper

def computeResiduals_reproj(params, cameraMatrices, distCoeffs, objPoints, imgPoints, foundCams):
    N_frames, N_cams, N_boardPoints, _ = imgPoints.shape
    params = params.reshape((N_cams-1,2,3,1))
    rvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,0]])
    tvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,1]])
    # params = params.reshape((N_cams,2,3,1))
    # rvecs = params[:,0]
    # tvecs = params[:,1]
    projMats = np.zeros((N_cams, 3, 4), dtype=np.float32)
    # distCoeffs = np.zeros((1,5), dtype=np.float32)
    imgPoints_undist = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan, dtype=np.float32)
    
    for cam in range(N_cams):
        R = cv2.Rodrigues(rvecs[cam])[0]
        projMats[cam] = cameraMatrices[cam] @ np.hstack([R,tvecs[cam]])
        imgPoints_undist[foundCams[:,cam], cam] = cv2.undistortImagePoints(imgPoints[foundCams[:,cam], cam].reshape(-1,2), cameraMatrices[cam], distCoeffs[cam]).reshape((-1,N_boardPoints,2))
    
    worldPoints = np.full((N_frames, N_boardPoints, 3), np.nan, dtype=np.float32)
    for frame in range(N_frames):
        cams = foundCams[frame]
        triangulatedPoints = np.full((N_boardPoints, 3), np.nan, dtype=np.float32)
        for boardPoint in range(N_boardPoints):
            triangulatedPoints[boardPoint] = triangulate(projMats[cams], imgPoints_undist[frame, cams, boardPoint])
        
        # R_obj, T_obj = rigidTransform(objPoints.T, triangulatedPoints.T)
        # worldPoints[frame] = objPoints @ R_obj.T + T_obj.T
        worldPoints[frame] = triangulatedPoints
    
    reprojectedPoints = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan)
    for cam in range(N_cams):
        frames_cam = foundCams[:,cam]
        N_frames_cam = np.sum(frames_cam)
        worldPoints_cam = worldPoints[frames_cam].reshape(N_frames_cam*N_boardPoints, 3)
        reprojectedPoints[frames_cam, cam] = cv2.projectPoints(worldPoints_cam, rvecs[cam], tvecs[cam], cameraMatrices[cam], distCoeffs[cam])[0].reshape(N_frames_cam,N_boardPoints,2)
    
    residuals = (reprojectedPoints[foundCams] - imgPoints[foundCams]).ravel()
    
    return residuals

def computeResiduals_reproj_int(params, objPoints, imgPoints, foundCams):
    N_frames, N_cams, N_boardPoints, _ = imgPoints.shape
    params = params.reshape((N_cams,15))
    rvecs = params[:,:3].reshape(N_cams,3,1)
    tvecs = params[:,3:6].reshape(N_cams,3,1)
    cameraMatrices = np.zeros((N_cams,3,3), dtype=np.float32)
    cameraMatrices[:,[0,1,0,1,2],[0,1,2,2,2]] = np.hstack([params[:,6:10],np.ones((N_cams,1))])
    distCoeffs = params[:,10:15]
    projMats = np.zeros((N_cams, 3, 4), dtype=np.float32)
    imgPoints_undist = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan, dtype=np.float32)
    
    for cam in range(N_cams):
        R = cv2.Rodrigues(rvecs[cam])[0]
        projMats[cam] = cameraMatrices[cam] @ np.hstack([R,tvecs[cam]])
        imgPoints_undist[foundCams[:,cam], cam] = cv2.undistortImagePoints(imgPoints[foundCams[:,cam], cam].reshape(-1,2), cameraMatrices[cam], distCoeffs[cam]).reshape((-1,N_boardPoints,2))
    
    worldPoints = np.full((N_frames, N_boardPoints, 3), np.nan, dtype=np.float32)
    for frame in range(N_frames):
        cams = foundCams[frame]
        triangulatedPoints = np.full((N_boardPoints, 3), np.nan, dtype=np.float32)
        for boardPoint in range(N_boardPoints):
            triangulatedPoints[boardPoint] = triangulate(projMats[cams], imgPoints_undist[frame, cams, boardPoint])
        
        R_obj, T_obj = rigidTransform(objPoints.T, triangulatedPoints.T)
        worldPoints[frame] = objPoints @ R_obj.T + T_obj.T
        # worldPoints[frame] = triangulatedPoints
    
    reprojectedPoints = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan)
    for cam in range(N_cams):
        frames_cam = foundCams[:,cam]
        N_frames_cam = np.sum(frames_cam)
        worldPoints_cam = worldPoints[frames_cam].reshape(N_frames_cam*N_boardPoints, 3)
        reprojectedPoints[frames_cam, cam] = cv2.projectPoints(worldPoints_cam, rvecs[cam], tvecs[cam], cameraMatrices[cam], distCoeffs[cam])[0].reshape(N_frames_cam,N_boardPoints,2)
    
    residuals = (reprojectedPoints[foundCams] - imgPoints[foundCams]).ravel()
    
    return residuals

def computeResiduals_pos(params, cameraMatrices, distCoeffs, objPoints, imgPoints, foundCams):
    N_frames, N_cams, N_boardPoints, _ = imgPoints.shape
    params = params.reshape((N_cams-1,2,3,1))
    # params = params.reshape((N_cams,2,3,1))
    rvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,0]])
    tvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,1]])
    # rvecs = params[:,0]
    # tvecs = params[:,1]
    projMats = np.zeros((N_cams, 3, 4), dtype=np.float32)
    imgPoints_undist = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan, dtype=np.float32)
    
    for cam in range(N_cams):
        R = cv2.Rodrigues(rvecs[cam])[0]
        projMats[cam] = cameraMatrices[cam] @ np.hstack([R,tvecs[cam]])
        imgPoints_undist[foundCams[:,cam], cam] = cv2.undistortImagePoints(imgPoints[foundCams[:,cam], cam].reshape(-1,2), cameraMatrices[cam], distCoeffs[cam]).reshape((-1,N_boardPoints,2))
    
    displacements = np.full((N_frames, N_boardPoints, 3), np.nan, dtype=np.float32)
    for frame in range(N_frames):
        cams = foundCams[frame]
        triangulatedPoints = np.full((N_boardPoints, 3), np.nan, dtype=np.float32)
        for boardPoint in range(N_boardPoints):
            triangulatedPoints[boardPoint] = triangulate(projMats[cams], imgPoints_undist[frame, cams, boardPoint])
        
        R_obj, T_obj = rigidTransform(objPoints.T, triangulatedPoints.T)
        displacements[frame] = (triangulatedPoints - objPoints @ R_obj.T - T_obj.T) @ R_obj  # get rid of last @ R_obj?
    
    residuals = displacements.ravel()
    
    return residuals

def computeResiduals_pos_int(params, objPoints, imgPoints, foundCams):
    N_frames, N_cams, N_boardPoints, _ = imgPoints.shape
    params = params.reshape((N_cams,15))
    rvecs = params[:,:3].reshape(N_cams,3,1)
    tvecs = params[:,3:6].reshape(N_cams,3,1)
    cameraMatrices = np.zeros((N_cams,3,3), dtype=np.float32)
    cameraMatrices[:,[0,1,0,1,2],[0,1,2,2,2]] = np.hstack([params[:,6:10],np.ones((N_cams,1))])
    distCoeffs = params[:,10:15]
    projMats = np.zeros((N_cams, 3, 4), dtype=np.float32)
    imgPoints_undist = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan, dtype=np.float32)
    
    for cam in range(N_cams):
        R = cv2.Rodrigues(rvecs[cam])[0]
        projMats[cam] = cameraMatrices[cam] @ np.hstack([R,tvecs[cam]])
        imgPoints_undist[foundCams[:,cam], cam] = cv2.undistortImagePoints(imgPoints[foundCams[:,cam], cam].reshape(-1,2), cameraMatrices[cam], distCoeffs[cam]).reshape((-1,N_boardPoints,2))
    
    displacements = np.full((N_frames, N_boardPoints, 3), np.nan, dtype=np.float32)
    for frame in range(N_frames):
        cams = foundCams[frame]
        triangulatedPoints = np.full((N_boardPoints, 3), np.nan, dtype=np.float32)
        for boardPoint in range(N_boardPoints):
            triangulatedPoints[boardPoint] = triangulate(projMats[cams], imgPoints_undist[frame, cams, boardPoint])
        
        R_obj, T_obj = rigidTransform(objPoints.T, triangulatedPoints.T)
        displacements[frame] = (triangulatedPoints - objPoints @ R_obj.T - T_obj.T) @ R_obj  # get rid of last @ R_obj?
    
    residuals = displacements.ravel()
    
    return residuals

def computeResiduals_pos_percam(params, cameraMatrices, distCoeffs, objPoints, imgPoints, foundCams):
    N_frames, N_cams, N_boardPoints, _ = imgPoints.shape
    params = params.reshape((N_cams-1,2,3,1))
    rvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,0]])
    tvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,1]])
    Rmats = np.full((N_cams,3,3), np.nan, dtype=np.float32)
    projMats = np.full((N_cams, 3, 4), np.nan, dtype=np.float32)
    imgPoints_undist = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan, dtype=np.float32)
    
    for cam in range(N_cams):
        Rmats[cam] = cv2.Rodrigues(rvecs[cam])[0]
        projMats[cam] = cameraMatrices[cam] @ np.hstack([Rmats[cam],tvecs[cam]])
        imgPoints_undist[foundCams[:,cam], cam] = cv2.undistortImagePoints(imgPoints[foundCams[:,cam], cam].reshape(-1,2), cameraMatrices[cam], distCoeffs[cam]).reshape((-1,N_boardPoints,2))
    
    displacements = np.full((N_frames, N_cams, N_boardPoints, 3), np.nan, dtype=np.float32)
    for frame in range(N_frames):
        cams = foundCams[frame]
        triangulatedPoints = np.full((N_boardPoints, 3), np.nan, dtype=np.float32)
        for boardPoint in range(N_boardPoints):
            triangulatedPoints[boardPoint] = triangulate(projMats[cams], imgPoints_undist[frame, cams, boardPoint])
        # R_obj, T_obj = rigidTransform(objPoints.T, triangulatedPoints.T)
        # extPoints = objPoints @ R_obj.T + T_obj.T
        
        for cam in np.where(cams)[0]:
            imgPoints_framecam = imgPoints_undist[frame,cam].reshape(-1,2)
            ret, rvec, tvec = cv2.solvePnP(objPoints, imgPoints_framecam, cameraMatrices[cam], None, flags=cv2.SOLVEPNP_SQPNP)  # see if there's a diff when using original imgpoints and adding distcoeffs here
            cv2.solvePnPRefineLM(objPoints, imgPoints_framecam, cameraMatrices[cam], None, rvec, tvec, criteria_refineLM)
            R_obj = cv2.Rodrigues(rvec)[0]
            pnpPoints = objPoints @ R_obj.T + tvec.T
            displacements[frame, cam] = pnpPoints - (triangulatedPoints @ Rmats[cam].T + tvecs[cam].T)
    
    residuals = displacements[foundCams].ravel()
    
    return residuals

def computeResiduals_pos_int_percam(params, objPoints, imgPoints, foundCams):
    N_frames, N_cams, N_boardPoints, _ = imgPoints.shape
    cameraMatrices = np.zeros((N_cams,3,3), dtype=np.float32)
    cameraMatrices[0,[0,1,0,1,2],[0,1,2,2,2]] = np.hstack([params[:4],1])
    distCoeffs = np.zeros((N_cams, 5), dtype=np.float32)
    distCoeffs[0] = params[4:9]
    params = params[9:].reshape((N_cams-1,15))
    rvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,:3].reshape(N_cams-1,3,1)])
    tvecs = np.vstack([np.zeros((1,3,1),dtype=np.float32), params[:,3:6].reshape(N_cams-1,3,1)])
    # rvecs = params[:,:3].reshape(N_cams,3,1)
    # tvecs = params[:,3:6].reshape(N_cams,3,1)
    cameraMatrices[1:,[0,1,0,1,2],[0,1,2,2,2]] = np.hstack([params[:,6:10],np.ones((N_cams-1,1))])
    distCoeffs[1:] = params[:,10:15]
    Rmats = np.full((N_cams,3,3), np.nan, dtype=np.float32)
    projMats = np.full((N_cams, 3, 4), np.nan, dtype=np.float32)
    imgPoints_undist = np.full((N_frames, N_cams, N_boardPoints, 2), np.nan, dtype=np.float32)
    
    for cam in range(N_cams):
        Rmats[cam] = cv2.Rodrigues(rvecs[cam])[0]
        projMats[cam] = cameraMatrices[cam] @ np.hstack([Rmats[cam],tvecs[cam]])
        imgPoints_undist[foundCams[:,cam], cam] = cv2.undistortImagePoints(imgPoints[foundCams[:,cam], cam].reshape(-1,2), cameraMatrices[cam], distCoeffs[cam]).reshape((-1,N_boardPoints,2))
    
    displacements = np.full((N_frames, N_cams, N_boardPoints, 3), np.nan, dtype=np.float32)
    for frame in range(N_frames):
        cams = foundCams[frame]
        triangulatedPoints = np.full((N_boardPoints, 3), np.nan, dtype=np.float32)
        for boardPoint in range(N_boardPoints):
            triangulatedPoints[boardPoint] = triangulate(projMats[cams], imgPoints_undist[frame, cams, boardPoint])
        # R_obj, T_obj = rigidTransform(objPoints.T, triangulatedPoints.T)
        # extPoints = objPoints @ R_obj.T + T_obj.T
        
        for cam in np.where(cams)[0]:
            imgPoints_framecam = imgPoints_undist[frame,cam].reshape(-1,2)
            ret, rvec, tvec = cv2.solvePnP(objPoints, imgPoints_framecam, cameraMatrices[cam], None, flags=cv2.SOLVEPNP_SQPNP)  # see if there's a diff when using original imgpoints and adding distcoeffs here
            cv2.solvePnPRefineLM(objPoints, imgPoints_framecam, cameraMatrices[cam], None, rvec, tvec, criteria_refineLM)
            R_obj = cv2.Rodrigues(rvec)[0]
            pnpPoints = objPoints @ R_obj.T + tvec.T
            displacements[frame, cam] = pnpPoints - (triangulatedPoints @ Rmats[cam].T + tvecs[cam].T)
    
    residuals = displacements[foundCams].ravel()
    
    return residuals

# %%
if __name__ == "__main__":
    charucoDetectors = updateDetectors(calibData)
    
    if INTRINSIC:
        if initCalib_path is None:
            imgPoints_df, objPoints_df = processVideos(videoPath_int, False)
            intrinsicErrors, perViewErrors = intrinsicCalib(imgPoints_df, objPoints_df)
            charucoDetectors = updateDetectors(calibData)
            for cam in cams:
                outliers = ~doubleMAD(perViewErrors[cam]).flatten()
                outliers = np.hstack([outliers, np.full(len(imgPoints_df)-len(outliers), False)])
                imgPoints_df.loc[outliers, cam] = np.nan
            print(f'1st pass intrinsic calibration error: {intrinsicErrors}')
        imgPoints_df, objPoints_df = processVideos(videoPath_int, False)
        intrinsicErrors, _ = intrinsicCalib(imgPoints_df, objPoints_df)
        charucoDetectors = updateDetectors(calibData)
        
        print(f'2nd pass intrinsic calibration error: {intrinsicErrors}')
    
    # %%
    if EXTRINSIC:
        imgPoints_df, objPoints_df = processVideos(videoPath_ext, True)
        extrinsicErrors = dict([(camPair, None) for camPair in camPairs])
        cam1 = cams[0]
        calibData.at[cam1, 'R'] = np.eye(3)
        calibData.at[cam1, 'T'] = np.zeros((3,1))
        for cam2 in cams[1:]:
            print(f'Starting extrinsic calibration for cam {cam2}')
            cameraMatrix_1, cameraMatrix_2 = calibData.loc[[cam1, cam2], "cameraMatrix"]
            distCoeffs_1, distCoeffs_2 = calibData.loc[[cam1, cam2], "distCoeffs"]
            ptIndex = imgPoints_df.loc[:, [cam1, cam2]].notna().all(axis=1)
            imgPoints_1, imgPoints_2 = list(imgPoints_df.loc[ptIndex, cam1]), list(imgPoints_df.loc[ptIndex, cam2])
            objPoints = list(objPoints_df.loc[ptIndex, cam1])
            
            error, _, _, _, _, R, T, _, _, _, _, perViewErrors = cv2.stereoCalibrateExtended(objPoints, imgPoints_1, imgPoints_2, cameraMatrix_1, distCoeffs_1, cameraMatrix_2, distCoeffs_2, imageSize, np.eye(3), np.zeros((3,1)))
            
            outliers = np.any(~doubleMAD(perViewErrors.flatten()).reshape(-1,2), axis=1).shape
            outliers = np.hstack([outliers, np.full(len(imgPoints_df)-len(outliers), False)])
            imgPoints_df.loc[outliers, cam2] = np.nan
            
            calibData.at[cam2, "R"] = R
            calibData.at[cam2, "T"] = T
            extrinsicErrors[cam1, cam2] = error
        
        print(f'Extrinsic calibration error: {extrinsicErrors}')
        
        # %%
        imgPoints_stereo = imgPoints_df.dropna(thresh=2)
        N_stereoFrames = len(imgPoints_stereo)
        found_array = imgPoints_stereo.notna().to_numpy()
        imgPoints_array = np.full((N_stereoFrames,N_cams,N_corners,2), np.nan, dtype=np.float32)
        if BUNDLEINT: params_initial = np.zeros((N_cams,15),dtype=np.float32)
        # if BUNDLEINT: params_initial = np.zeros((N_cams,9),dtype=np.float32)
        else: params_initial = np.zeros((N_cams,2,3,1),dtype=np.float32)
        for camIndex, cam in enumerate(cams):
            frames = found_array[:,camIndex]
            cameraMatrix, distCoeffs, R, T = calibData.loc[cam]
            imgPoints_array[frames, camIndex] = np.stack(imgPoints_stereo.loc[frames,cam]).reshape((-1,N_corners,2))
            if BUNDLEINT: 
                params_initial[camIndex, :3] = cv2.Rodrigues(R)[0].T
                params_initial[camIndex, 3:6] = T.T
                params_initial[camIndex, 6:10] = cameraMatrix[[0,1,0,1],[0,1,2,2]]
                params_initial[camIndex, 10:] = distCoeffs
                # params_initial[camIndex, :4] = cameraMatrix[[0,1,0,1],[0,1,2,2]]
                # params_initial[camIndex, 4:] = distCoeffs
                
            else: 
                # imgPoints_array[frames, camIndex] = cv2.undistortImagePoints(np.vstack(imgPoints_stereo.loc[frames,cam]), cameraMatrix, distCoeffs).reshape((-1,N_corners,2))
                params_initial[camIndex, 0] = cv2.Rodrigues(R)[0]
                params_initial[camIndex, 1] = T
        
        
        if BUNDLEINT: 
            residualFunction = computeResiduals_pos_int_percam if MINPOS else computeResiduals_reproj_int
            
        else: 
            # params_initial[:, 1] = np.stack(calibData.loc[:,"T"])
            cameraMatrices = np.stack(calibData.loc[:,"cameraMatrix"], dtype=np.float32)
            distCoeffs = np.stack(calibData.loc[:,"distCoeffs"], dtype=np.float32)
            residualFunction = computeResiduals_pos_percam if MINPOS else computeResiduals_reproj
        # params_initial = params_initial[1:].ravel()
        params_initial = params_initial.ravel()[6:]
        if TAGBASED: objPoints = np.array(charucoBoard.getObjPoints()).reshape(-1,3)
        else: objPoints = charucoBoard.getChessboardCorners()
        
        # Jacobian sparsity matrix
        # N_residuals = np.sum(found_array) * N_charucoCorners * 2
        # N_params = (N_cams-1) * 6
        # jac_sparsity = lil_matrix((N_residuals, N_params), dtype=int)
        # nextResidual = 0
        # for frame in range(N_stereoFrames):
        #     N_residuals_frame = np.sum(found_array[frame]) * N_charucoCorners * 2
        #     cams_frame = np.where(found_array[frame])[0]
        #     for cam in cams_frame:
        #         if cam == 0: continue
        #         jac_sparsity[nextResidual:nextResidual+N_residuals_frame, (cam-1)*6:(cam-1)*6+6] = 1
        #     nextResidual = nextResidual+N_residuals_frame
        
        # Removal of outliers
        if BUNDLEINT: residuals_initial = residualFunction(params_initial, objPoints, imgPoints_array, found_array)
        else: residuals_initial = residualFunction(params_initial, cameraMatrices, distCoeffs, objPoints, imgPoints_array, found_array)
        rmse_initial = np.sqrt(np.mean(residuals_initial**2))
        print(f'Extrinsic RMSE with outliers: {rmse_initial}')
        foundFrames = np.where(found_array)[0]  # Frame indices of all observations
        N_observations = len(foundFrames)  # Each view of the board by a camera is an observation (so each frame can have 2-5 observations)
        # outliers = np.where(np.abs(residuals_initial)>2*np.std(residuals_initial))[0]  # Indices of outliers in the full vector of residuals
        residualNorms = np.linalg.norm(residuals_initial.reshape(-1, 3 if MINPOS else 2), axis=1)
        outliers = np.where(~doubleMAD(residualNorms))
        outliers_obs = np.unravel_index(outliers, (N_observations,N_corners))[0]  # Observation indices of each outlier
        outliers_frames = foundFrames[outliers_obs]  # Frame indices of each outlier
        # if MINPOS: 
        #     # outliers_frames = np.unravel_index(outliers, (N_stereoFrames,N_corners,3))[0]  # Observation indices of each outlier
        #     outliers_obs = np.unravel_index(outliers, (N_observations,N_corners,3))[0]  # Observation indices of each outlier
        #     outliers_frames = foundFrames[outliers_obs]  # Frame indices of each outlier
        # else: 
        #     # outliers_obs = np.unravel_index(outliers, (N_observations,N_corners,2))[0]  # Observation indices of each outlier
        #     outliers_obs = np.unravel_index(outliers, (N_observations,N_corners,2))[0]  # Observation indices of each outlier
        #     outliers_frames = foundFrames[outliers_obs]  # Frame indices of each outlier
        imgPoints_array = np.delete(imgPoints_array, outliers_frames, axis=0)
        found_array = np.delete(found_array, outliers_frames, axis=0)
        if BUNDLEINT: residuals_filtered = residualFunction(params_initial, objPoints, imgPoints_array, found_array)
        else: residuals_filtered = residualFunction(params_initial, cameraMatrices, distCoeffs, objPoints, imgPoints_array, found_array)
        rmse_filtered = np.sqrt(np.mean(residuals_filtered**2))
        print(f'Extrinsic RMSE without outliers: {rmse_filtered}')

        # %%
        # Optimization
        print('Starting bundle adjustment.')
        if BUNDLEINT: lsqResult = least_squares(residualFunction, params_initial, diff_step=0.03, max_nfev=10, verbose=2, x_scale='jac', method='trf', loss='linear', args=(objPoints, imgPoints_array, found_array))
        else: lsqResult = least_squares(residualFunction, params_initial, diff_step=0.03, max_nfev=10, verbose=2, x_scale='jac', method='trf', loss='linear', args=(cameraMatrices, distCoeffs, objPoints, imgPoints_array, found_array))
        residuals_final = lsqResult.fun
        rmse_final = np.sqrt(np.mean(residuals_final**2))
        print(f'Bundle adjustment final RMSE: {rmse_final}')
        
        # %%
        if BUNDLEINT:
            # params_final = lsqResult.x.reshape(N_cams,15)
            # params_final = lsqResult.x.reshape(N_cams,9)
            params_final = lsqResult.x
            calibData.at[cams[0], "cameraMatrix"][[0,1,0,1],[0,1,2,2]] = params_final[:4]
            calibData.at[cams[0], "distCoeffs"] = params_final[4:9].reshape(1,-1)
            params_final = params_final[9:].reshape(N_cams-1, 15)
            for camIndex in range(1,N_cams):
                calibData.at[cams[camIndex], "R"] = cv2.Rodrigues(params_final[camIndex-1, :3])[0]
                calibData.at[cams[camIndex], "T"] = params_final[camIndex-1, 3:6].reshape(3,1)
                calibData.at[cams[camIndex], "cameraMatrix"][[0,1,0,1],[0,1,2,2]] = params_final[camIndex-1, 6:10]
                calibData.at[cams[camIndex], "distCoeffs"] = params_final[camIndex-1, 10:].reshape(1,-1)
                # calibData.at[cams[camIndex], "cameraMatrix"][[0,1,0,1],[0,1,2,2]] = params_final[camIndex, :4]
                # calibData.at[cams[camIndex], "distCoeffs"] = params_final[camIndex, 4:].reshape(1,-1)
        else:
            params_final = lsqResult.x.reshape(N_cams-1,2,3,1)
            # params_final = lsqResult.x.reshape(N_cams,2,3,1)
            # for camIndex in range(1,N_cams):
            for camIndex in range(1,N_cams):
                calibData.at[cams[camIndex], "R"] = cv2.Rodrigues(params_final[camIndex-1, 0])[0]
                calibData.at[cams[camIndex], "T"] = params_final[camIndex-1, 1]
                # calibData.at[cams[camIndex], "R"] = cv2.Rodrigues(params_final[camIndex, 0])[0]
                # calibData.at[cams[camIndex], "T"] = params_final[camIndex, 1]
    
    # %%   
    calibData.reset_index(names='id').to_json(f'calib_files/camera/{time.strftime("%Y-%m-%d_%H_%M_%S")}.json', orient='records', indent=3)
# %%
