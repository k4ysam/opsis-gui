# %%
import numpy as np
import cv2
from scipy.optimize import least_squares
import pandas as pd
import time
import multiprocessing as mp
from pynput import keyboard
from scipy.sparse import lil_matrix
import itertools as it

TARGET = True  # True for target calibration (target origin at charuco origin), False for reference calibration (origin at top tag)
LIVE = False
MINPOS = False  # Minimizes positional error if True, otherwise reprojection error is minimized
# TAGVECTORS = False

cams = [1,2,3,4,5]  # List of cameras to calibrate, all of which must be included in camList
captureRate = 60  # fps at which frames are synchronously grabbed from all cameras for calibration
frameSkip = 1  # skip calibration of every N frames captured

initBoard_path = 'calib_files/tool/dodecaBoard_target.txt'  # Board points file to use for initial values
cameraCalib_path = 'calib_files/camera/T33_minpos_int_percam_cam1fixed.json'  # Calibration file to use for initial values; can only be set to None if INTRINSIC is True
video_path = 'calib_videos/targetcalib_T33'  # folder to grab video from if VIDEO is true but LIVE is false

maxIter_preop = 100
minDiff_preop = 0.0001
outlierThreshold = 2  # Threshold for outlier rejection after pre-optimization, in standard deviations
maxIter_lsq = 10

# Camera calibration import
calibData = pd.read_json(cameraCalib_path, orient='records')
for i, id in enumerate(calibData.loc[:,'id']): 
    calibData.at[i,'id'] = tuple(id) if type(id) == list else id
calibData = calibData.set_index('id')
calibData = calibData.applymap(np.array)
calibData = calibData.replace(np.nan, None)
N_cams = len(cams)
cameraMatrices = np.array(calibData.loc[cams, 'cameraMatrix'])
distCoeffs = np.array(calibData.loc[cams, 'distCoeffs'])
projMats = np.zeros((N_cams, 3, 4), dtype=np.float32)  # projective matrices that transform 3D points in world frame to image points in each camera
for camIndex, cam in enumerate(cams):
    _, _, R, T = calibData.loc[cam]
    projMats[camIndex] = cameraMatrices[camIndex] @ np.hstack([R,T])

# Aruco marker params
markerSize = 22 if TARGET else 33  # in mm
N_tags = 11
firstTagId = 0 if TARGET else 11
tagBoardPoints = np.loadtxt(initBoard_path, dtype=np.float32).reshape(-1,4,3)
tagsDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
detectorParams = cv2.aruco.DetectorParameters()
detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # Default CORNER_REFINE_NONE
detectorParams.cornerRefinementMaxIterations = 1000  # Default 30
detectorParams.cornerRefinementMinAccuracy = 0.001  # Default 0.1
detectorParams.adaptiveThreshWinSizeStep = 2  # Default 10
detectorParams.adaptiveThreshWinSizeMax = 15  # Default 23
detectorParams.adaptiveThreshConstant = 8  # Default 7
arucoDetector = cv2.aruco.ArucoDetector(tagsDict, detectorParams)
tagsBoard = cv2.aruco.Board(tagBoardPoints, tagsDict, np.arange(N_tags)+firstTagId)
criteria_refineLM = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 200, 1.19209e-07)  # Default count = 20, epsilon = 1.19209e-07

tagPoints = np.array([[markerSize/2, -markerSize/2, 0], 
                    [-markerSize/2, -markerSize/2, 0], 
                    [-markerSize/2, markerSize/2, 0], 
                    [markerSize/2, markerSize/2, 0]], dtype=np.float32)

if TARGET:
    # Charuco board params
    patternSize = (6, 8)  # number of squares in X, Y
    minPoints = 35  # Minimum number of visible board points to use a frame for calibration
    squareLength = 32.004  # in mm
    markerLength = 17.992  # in mm
    charucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    patternSize_chess = (patternSize[1]-1, patternSize[0]-1)  # Used for sharpness detection
    N_charucoCorners = (patternSize[0]-1) * (patternSize[1]-1)  # Total number of charuco corners, used to verify full board visibility
    charucoBoard = cv2.aruco.CharucoBoard(patternSize, squareLength, markerLength, charucoDict)
    charucoObjPoints = charucoBoard.getChessboardCorners()

    charucoDetectors = dict()
    for cam in cams:
        charucoParams = cv2.aruco.CharucoParameters()
        charucoParams.tryRefineMarkers = False  # Default False
        charucoParams.cameraMatrix = calibData.at[cam, "cameraMatrix"]
        charucoParams.distCoeffs = calibData.at[cam, "distCoeffs"]
        charucoParams.distCoeffs = np.zeros((1,5), dtype=np.float32)
        charucoDetectors[cam] = cv2.aruco.CharucoDetector(charucoBoard, charucoParams, detectorParams)

def runCam(cam, cameraMatrix, distCoeffs, childConn, stopEvent, barrier):
    if TARGET: charucoDetector = charucoDetectors[cam]
    if LIVE: cap = cv2.VideoCapture(f"udpsrc address=192.168.3.2 port=500{cam} ! application/x-rtp, clock-rate=90000, payload=96 ! rtph264depay ! h264parse ! avdec_h264 discard-corrupted-frames=true skip-frame=1 ! videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true sync=false", cv2.CAP_GSTREAMER)
    else: cap = cv2.VideoCapture(f'{video_path}/{cam}.mp4')
    if not cap.isOpened():
        print(f"Cannot open camera {cam}.")
        return
    
    frameIndex = 0
    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        if cv2.pollKey() == ord('q'):
            cv2.destroyWindow(f'Camera {cam}')
            stopEvent.set()
            break
        
        barrier.wait()
        ret, frame = cap.read()  # ret is True if frame is read correctly
        
        if not ret:
            if not LIVE and not stopEvent.is_set():
                print("Prerecorded video finished.")
                stopEvent.set()
            else:
                print(f"Can't receive frame from camera {cam}.")
            childConn.send([None]*4)
            continue
        
        message = [None] * 4
        # frame = cv2.undistort(frame, cameraMatrix, distCoeffs)
        if frameIndex % frameSkip == 0:
            tagCorners, tagIds, _ = arucoDetector.detectMarkers(frame)
            
            if TARGET: charucoCorners, charucoIds, charucoMarkerCorners, charucoMarkerIds = charucoDetector.detectBoard(frame)
            
            if TARGET and charucoCorners is not None:
                # frame = cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds)
                frame = cv2.aruco.drawDetectedMarkers(frame, charucoMarkerCorners, charucoMarkerIds)
                if len(charucoIds) >= minPoints:
                    # charucoCorners_array = np.array(charucoCorners, dtype=np.float32).reshape(-1,2)
                    # charucoCorners_undist = cv2.undistortImagePoints(charucoCorners_array, cameraMatrix, distCoeffs)
                    message[:2] = charucoCorners, charucoIds
                
            if tagIds is not None:
                frame = cv2.aruco.drawDetectedMarkers(frame, tagCorners, tagIds)
                tagIds_filtered = np.all([tagIds>=firstTagId, tagIds<firstTagId+N_tags], axis=0)
                if np.sum(tagIds_filtered) > 0:
                    tagIds = tagIds[tagIds_filtered]
                    tagCorners = np.array(tagCorners)[tagIds_filtered]
                    # tagCorners_array = np.array(tagCorners, dtype=np.float32).reshape(-1,2)
                    # tagCorners_undist = cv2.undistortImagePoints(tagCorners_array, cameraMatrix, distCoeffs).reshape(-1,4,2)
                    message[2:] = tagCorners, tagIds
        
        childConn.send(message)
        
        cv2.imshow(f'Camera {cam}', frame)
        frameIndex += 1

    cap.release()
    childConn.close()
    cv2.destroyWindow(f'Camera {cam}')

def onPress(key):
    global captureEvent, stopEvent
    try:
        if key.char == 'q':
            stopEvent.set()
            print("\nStopping calibration.")
    except AttributeError:
        
        pass

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

def rigidRotate(A, B, repeats=None):
    # Returns the R which rotates the points in A towards those in B while maintaining their distance to the origin
    # A and B must be of shape 3xN
    
    if repeats is not None:
        A = np.repeat(A, repeats, axis=1)
        B = np.repeat(B, repeats, axis=1)
    
    H = A @ B.T
    U, S, Vh = np.linalg.svd(H)
    
    R = Vh.T @ U.T
    if np.linalg.det(R) < 0:
        Vh[2,:] *= -1
        R = Vh.T @ U.T
    
    return R

def computeResiduals_proj(params, cameraMatrices, imgPoints, worldPoints, foundTags):
    N_frames, N_tags, N_cams, _, _ = imgPoints.shape
    # if TAGVECTORS:
    #     tagVectors = params.reshape(N_tags, 2, 3)
    #     objPoints = np.full((N_tags, 4, 3), np.nan, dtype=np.float32)
    #     for tag in range(N_tags):
    #         rvec, tvec = tagVectors[tag]
    #         R = cv2.Rodrigues(rvec)[0]
    #         objPoints[tag] = tagPoints @ R.T + tvec
    if TARGET:
        objPoints = params[:N_tags*12].reshape(N_tags, 4, 3)
        tipPositions = params[N_tags*12:].reshape(N_cams, 3, 1)
        # tipPositions_img = np.array([cv2.projectPoints(tipPositions[i], np.zeros(3), np.zeros(3), cameraMatrices[i], None)[0][0,0] for i in range(N_cams)])
    else:
        objPoints = params.reshape(N_tags, 4, 3)

    worldPoints_obj = np.full((N_frames, N_tags, N_cams, 4, 3), np.nan, dtype=np.float32)
    # if TARGET: worldPoints_tips = np.full((N_frames, N_cams, 3), np.nan, dtype=np.float32)
    for frame, camIndex in it.product(range(N_frames), range(N_cams)):
        tags = np.where(foundTags[frame, :, camIndex])[0]
        N_tags_frame = len(tags)
        if N_tags_frame == 0: continue
        # if TARGET:
        #     objPoints_frame = np.vstack([objPoints[tags].reshape(N_tags_frame*4,3), [0.,0.,0.]])
        #     imgPoints_frame = np.vstack([imgPoints[frame, tags, camIndex].reshape(N_tags_frame*4,2), tipPositions_img[camIndex]])
        worldPoints_frame = worldPoints[frame, tags, camIndex].reshape(N_tags_frame*4,3)
        objPoints_frame = objPoints[tags].reshape(N_tags_frame*4,3)
        imgPoints_frame = imgPoints[frame, tags, camIndex].reshape(N_tags_frame*4,2)
        
        if TARGET:
            tipPos = tipPositions[camIndex]
            R_obj = rigidRotate(objPoints_frame.T, worldPoints_frame.T - tipPos)
            worldPoints_obj[frame, tags, camIndex] = (objPoints_frame @ R_obj.T + tipPos.T).reshape(N_tags_frame,4,3)
            # ret, rvec, T_obj = cv2.solvePnP(objPoints_frame, imgPoints_frame, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_SQPNP)
            # if not ret: continue
            # cv2.solvePnPRefineLM(objPoints_frame, imgPoints_frame, cameraMatrices[camIndex], None, rvec, T_obj, criteria_refineLM)
            # R_obj = cv2.Rodrigues(rvec)[0]
            # worldPoints_obj[frame, tags, camIndex] = (objPoints_frame @ R_obj.T + tipPositions[camIndex].T).reshape(N_tags_frame,4,3)
            # worldPoints_tips[frame, camIndex] = objPoints_frame[-1] @ R_obj.T + T_obj.T
        else:
            # R_obj, T_obj = rigidTransform(objPoints_frame.T, worldPoints_frame.T)
            ret, rvec, T_obj = cv2.solvePnP(objPoints_frame, imgPoints_frame, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_SQPNP)
            if not ret: continue
            cv2.solvePnPRefineLM(objPoints_frame, imgPoints_frame, cameraMatrices[camIndex], None, rvec, T_obj, criteria_refineLM)
            R_obj = cv2.Rodrigues(rvec)[0]
            worldPoints_obj[frame, tags, camIndex] = (objPoints_frame @ R_obj.T + T_obj.T).reshape(N_tags_frame,4,3)
    
    reprojectedPoints = np.full((N_frames, N_tags, N_cams, 4, 2), np.nan)
    for camIndex in range(N_cams):
        allTags_cam = foundTags[:,:,camIndex]
        N_tags_cam = np.sum(allTags_cam)
        
        worldPoints_cam = worldPoints_obj[allTags_cam, camIndex].reshape(N_tags_cam*4, 3)
        reprojectedPoints_homogeneous = worldPoints_cam @ cameraMatrices[camIndex].T
        reprojectedPoints[allTags_cam, camIndex] = (reprojectedPoints_homogeneous[:,:2] / reprojectedPoints_homogeneous[:,[2]]).reshape(N_tags_cam,4,2)
    
    # if TARGET:
    #     foundCams = np.any(foundTags, axis=1)
    #     tipResiduals = np.full((N_frames, N_cams, 2), np.nan)
    #     for camIndex in range(N_cams):
    #         frames = foundCams[:,camIndex]
    #         reprojectedTips = cv2.projectPoints(worldPoints_tips[frames, camIndex], np.zeros(3), np.zeros(3), cameraMatrices[camIndex], None)[0].reshape(-1,2)
    #         tipResiduals[frames, camIndex] = reprojectedTips - tipPositions_img[camIndex]
        
    #     residuals = np.vstack([(reprojectedPoints[foundTags] - imgPoints[foundTags]).reshape(-1,2), tipResiduals[foundCams]]).ravel()
    
    residuals = (reprojectedPoints[foundTags] - imgPoints[foundTags]).ravel()
    
    return residuals

def computeResiduals_disp(params, worldPoints, foundTags):
    N_frames, N_tags, N_cams, _, _ = worldPoints.shape
    # if TAGVECTORS:
    #     tagVectors = params.reshape(N_tags, 2, 3)
    #     objPoints = np.full((N_tags, 4, 3), np.nan, dtype=np.float32)
    #     for tag in range(N_tags):
    #         rvec, tvec = tagVectors[tag]
    #         R = cv2.Rodrigues(rvec)[0]
    #         objPoints[tag] = tagPoints @ R.T + tvec
    if TARGET:
        objPoints = params[:N_tags*12].reshape(N_tags, 4, 3)
        tipPositions = params[N_tags*12:].reshape(N_cams, 3, 1)
    else:
        objPoints = params.reshape(N_tags, 4, 3)

    displacements = np.full((N_tagFrames, N_tags, N_cams, 4, 3), np.nan, dtype=np.float32)
    for frame, camIndex in it.product(range(N_frames), range(N_cams)):
        tags = np.where(foundTags[frame, :, camIndex])[0]
        N_tags_frame = len(tags)
        if N_tags_frame == 0: continue
        objPoints_frame = objPoints[tags].reshape(N_tags_frame*4,3)
        worldPoints_frame = worldPoints[frame, tags, camIndex].reshape(N_tags_frame*4,3)
        
        if TARGET:
            tipPos = tipPositions[camIndex]
            R_obj = rigidRotate(objPoints_frame.T, worldPoints_frame.T - tipPos)
            displacements[frame, tags, camIndex] = ((objPoints_frame @ R_obj.T + tipPos.T - worldPoints_frame) @ R_obj).reshape(N_tags_frame,4,3)
        else:
            R_obj, T_obj = rigidTransform(objPoints_frame.T, worldPoints_frame.T)
            displacements[frame, tags, camIndex] = ((objPoints_frame @ R_obj.T + T_obj.T - worldPoints_frame) @ R_obj).reshape(N_tags_frame,4,3)
    
    residuals = displacements[foundTags].ravel()
    
    return residuals

def tipError(objPoints, imgPoints, worldPoints, foundTags, tipPositions):
    N_frames, N_tags, N_cams, _, _ = worldPoints.shape
    displacements = np.full((N_frames, N_cams, 3), np.nan, dtype=np.float32)
    for frame, camIndex in it.product(range(N_frames), range(N_cams)):
        tags = np.where(foundTags[frame, :, camIndex])[0]
        N_tags_frame = len(tags)
        if N_tags_frame == 0: continue
        objPoints_frame = objPoints[tags].reshape(N_tags_frame*4,3)
        worldPoints_frame = worldPoints[frame, tags, camIndex].reshape(N_tags_frame*4,3)
        tipPos = tipPositions[camIndex]
        imgPoints_frame = imgPoints[frame, tags, camIndex].reshape(N_tags_frame*4,2)
        
        # R_obj, T_obj = rigidTransform(objPoints_frame.T, worldPoints_frame.T)
        ret, rvec, T_obj = cv2.solvePnP(objPoints_frame, imgPoints_frame, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_SQPNP)
        if not ret: continue
        cv2.solvePnPRefineLM(objPoints_frame, imgPoints_frame, cameraMatrices[camIndex], None, rvec, T_obj, criteria_refineLM)
        R_obj = cv2.Rodrigues(rvec)[0]
        displacements[frame, camIndex] = (R_obj.T @ (T_obj-tipPos)).T
    foundCams = np.any(foundTags, axis=1)
    rmse_perAxis = np.sqrt(np.mean(displacements[foundCams]**2, axis=0))
    errorLengths = np.linalg.norm(displacements[foundCams], axis=1, keepdims=True)
    rmse_euclidian = np.sqrt(np.mean(errorLengths**2))
    return rmse_perAxis, rmse_euclidian

def positionalError(objPoints, worldPoints, foundTags, tipPositions=None):
    if tipPositions is not None: params = np.hstack([objPoints.ravel(), tipPositions.ravel()])
    else: params = objPoints.ravel()
    residuals = computeResiduals_disp(params, worldPoints, foundTags).reshape((-1,3))
    rmse_perAxis = np.sqrt(np.mean(residuals**2, axis=0))
    errorLengths = np.linalg.norm(residuals, axis=1, keepdims=True)
    rmse_euclidian = np.sqrt(np.mean(errorLengths**2))
    return rmse_perAxis, rmse_euclidian

def reprojectionError(objPoints, cameraMatrices, imgPoints, worldPoints, foundTags, tipPositions=None):
    if tipPositions is not None: params = np.hstack([objPoints.ravel(), tipPositions.ravel()])
    else: params = objPoints.ravel()
    residuals = computeResiduals_proj(params, cameraMatrices, imgPoints, worldPoints, foundTags).reshape((-1,2))
    rmse_perAxis = np.sqrt(np.mean(residuals**2, axis=0))
    errorLengths = np.linalg.norm(residuals, axis=1, keepdims=True)
    rmse_euclidian = np.sqrt(np.mean(errorLengths**2))
    return rmse_perAxis, rmse_euclidian

def reportError(objPoints, cameraMatrices, imgPoints, worldPoints, foundTags, tipPositions=None):
    reprojError_ax_init, reprojError_euc_init = reprojectionError(objPoints, cameraMatrices, imgPoints, worldPoints, foundTags, tipPositions)
    print(f'Reprojection error: {reprojError_ax_init} per axis, {reprojError_euc_init} euclidian')
    posError_ax_init, posError_euc_init = positionalError(objPoints, worldPoints, foundTags, tipPositions)
    print(f'Positional error: {posError_ax_init} per axis, {posError_euc_init} euclidian')
    if tipPositions is not None:
        tipError_ax_init, tipError_euc_init = tipError(objPoints, imgPoints, worldPoints, foundTags, tipPositions)
        print(f'Tip error: {tipError_ax_init} per axis, {tipError_euc_init} euclidian')

# %%
if __name__ == "__main__":
    processes = dict()
    parentConns = dict()
    childConns = dict()
    barrier = mp.Barrier(len(cams)+1, timeout=10)
    stopEvent = mp.Event()
    listener = keyboard.Listener(on_press=onPress)
    listener.start()
    
    for camIndex, cam in enumerate(cams):
        parentConn, childConn = mp.Pipe(False)  # True for two-way communication, False for one-way
        process = mp.Process(target=runCam, args=(cam, cameraMatrices[camIndex], distCoeffs[camIndex], childConn, stopEvent, barrier))
        process.start()
        processes[cam] = process
        parentConns[cam] = parentConn
        childConns[cam] = childConn
    
    capTiming = 1/captureRate
    capIndex_charuco = 0  # current index of simultaneous captures where at least one camera found the board
    capIndex_tags = 0  # current index of simultaneous captures where at least one camera found the board
    lastCapTime = -capTiming
    if TARGET: charucoImgPoints_dict = dict([(cam, [None]*500) for cam in cams])  # Each simultaneous capture appends an inner list including numpy arrays of corners found by each camera, or None for any that didn't see it
    tagImgPoints_dict = dict([(cam, [None]*500) for cam in cams])

    while (not stopEvent.is_set()) or barrier.n_waiting != 0:
        if time.time() - lastCapTime >= capTiming:
            lastCapTime = time.time()
            barrier.wait()
            if TARGET: charucoFound = False
            tagFound = False
            
            for cam in cams:
                charucoCorners, charucoIds, tagCorners, tagIds = parentConns[cam].recv()
                if charucoCorners is not None:
                    # objPoints, imgPoints = charucoBoard.matchImagePoints(charucoCorners, charucoIds)
                    charucoImgPoints_dict[cam][capIndex_charuco] = charucoCorners
                    charucoFound = True
                
                if tagCorners is not None:
                    tagCorners_array = np.full((N_tags, 4, 2), np.nan, dtype=np.float32)
                    tagCorners_array[tagIds-firstTagId] = tagCorners
                    tagImgPoints_dict[cam][capIndex_tags] = tagCorners_array
                    tagFound = True
            
            if TARGET and charucoFound: 
                capIndex_charuco += 1
                if capIndex_charuco % 500 == 0: 
                    for cam in cams: charucoImgPoints_dict[cam] += [None]*500
            
            if tagFound: 
                capIndex_tags += 1
                if capIndex_tags % 500 == 0: 
                    for cam in cams: tagImgPoints_dict[cam] += [None]*500

# %%
    if TARGET:
        # Finding average charuco origin
        charucoImgPoints_df = pd.DataFrame(charucoImgPoints_dict, columns=cams).dropna(thresh=1)
        N_charucoFrames = len(charucoImgPoints_df)
        charuco_foundCams = charucoImgPoints_df.notna().to_numpy()
        charucoImgPoints = np.full((N_charucoFrames, N_cams, N_charucoCorners, 2), np.nan, dtype=np.float32)
        for camIndex, cam in enumerate(cams):
            frames = charuco_foundCams[:,camIndex]
            N_frames = np.sum(frames)
            if N_frames == 0: continue
            charucoImgPoints[frames, camIndex] = np.stack(charucoImgPoints_df.loc[frames,cam]).reshape(N_frames,N_charucoCorners,2)
        
        charucoOrigins = np.full((N_cams, 3, 1), np.nan, dtype=np.float32)
        for camIndex in range(N_cams):
            charucoOrigins_cam = np.full((3, N_charucoFrames), np.nan, dtype=np.float32)
            frames = charuco_foundCams[:,camIndex]
            for frame in range(N_charucoFrames):
                if not charuco_foundCams[frame, camIndex]: continue
                ret, rvec, tvec = cv2.solvePnP(charucoObjPoints, charucoImgPoints[frame, camIndex], cameraMatrices[camIndex], distCoeffs[camIndex], flags=cv2.SOLVEPNP_SQPNP)
                if not ret: continue
                cv2.solvePnPRefineLM(charucoObjPoints, charucoImgPoints[frame, camIndex], cameraMatrices[camIndex], distCoeffs[camIndex], rvec, tvec, criteria_refineLM)
                charucoOrigins_cam[:, [frame]] = tvec
            charucoOrigins[camIndex] = np.mean(charucoOrigins_cam[:,frames], axis=1, keepdims=True)
        
    # Filtering & organizing tag data
    tagImgPoints_df = pd.DataFrame(tagImgPoints_dict, columns=cams).dropna(thresh=1)
    N_tagFrames = len(tagImgPoints_df)
    tags_foundCams = tagImgPoints_df.notna().to_numpy()  # Boolean array of cams that see tags at each frame
    tagImgPoints = np.full((N_tagFrames, N_tags, N_cams, 4, 2), np.nan, dtype=np.float32)
    for camIndex, cam in enumerate(cams):
        frames = tags_foundCams[:,camIndex]
        # cameraMatrix, distCoeffs, R, T = calibData.loc[cam]
        # tagImgPoints[frames, :, camIndex] = np.stack(tagImgPoints_df.loc[frames,cam])
        tagImgPoints[frames, :, camIndex] = cv2.undistortImagePoints(np.vstack(tagImgPoints_df.loc[frames,cam]).reshape(-1,2), cameraMatrices[camIndex], distCoeffs[camIndex]).reshape((-1,N_tags,4,2))
        
    foundTags = ~np.isnan(tagImgPoints[:,:,:,0,0])  # Boolean array of visible tags at each frame for each camera
    print(f'Frames per tag: {np.sum(foundTags, axis=(0,2))}')
    
    tagWorldPoints = np.full((N_tagFrames, N_tags, N_cams, 4, 3), np.nan, dtype=np.float32)
    for frame, tag, camIndex in zip(*np.where(foundTags)):
        imgPoints = tagImgPoints[frame, tag, camIndex]
        ret, rvec, tvec = cv2.solvePnP(tagPoints, imgPoints, cameraMatrices[camIndex], None, flags=cv2.SOLVEPNP_SQPNP)
        if not ret: continue
        cv2.solvePnPRefineLM(tagPoints, imgPoints, cameraMatrices[camIndex], None, rvec, tvec, criteria_refineLM)
        tagWorldPoints[frame, tag, camIndex] = tagPoints @ cv2.Rodrigues(rvec)[0].T + tvec.T
        
    # Initial error metrics
    reportError(tagBoardPoints, cameraMatrices, tagImgPoints, tagWorldPoints, foundTags, charucoOrigins if TARGET else None)
    
    # Adding displacement to charuco origin as offset to objPoints to minimize bias before bundle adjustment
    # displacements = np.full((N_tagFrames, 3), np.nan, dtype=np.float32)
    iter_preop = 0
    offsets = []
    while True:
        displacements = np.full((N_tagFrames, N_tags, N_cams, 4, 3), np.nan, dtype=np.float32)
        for frame, camIndex in it.product(range(N_tagFrames), range(N_cams)):
            tags = np.where(foundTags[frame, :, camIndex])[0]
            N_tags_frame = len(tags)
            if N_tags_frame == 0: continue
            objPoints = tagBoardPoints[tags].reshape(N_tags_frame*4,3)
            worldPoints = tagWorldPoints[frame, tags, camIndex].reshape(N_tags_frame*4,3)
            imgPoints = tagImgPoints[frame, tags, camIndex].reshape(N_tags_frame*4,2)
            
            if TARGET:
                R_obj = rigidRotate(objPoints.T, worldPoints.T - charucoOrigins[camIndex])
                displacements[frame,tags,camIndex] = ((objPoints @ R_obj.T + charucoOrigins[camIndex].T - worldPoints) @ R_obj).reshape(N_tags_frame,4,3)
            else:
                R_obj, T_obj = rigidTransform(objPoints.T, worldPoints.T)
                displacements[frame,tags,camIndex] = ((objPoints @ R_obj.T + T_obj.T - worldPoints) @ R_obj).reshape(N_tags_frame,4,3)
            
        offset = np.stack([-np.mean(displacements[:,tag][foundTags[:,tag]], axis=0) for tag in range(N_tags)])
        tagBoardPoints += offset
        
        iter_preop += 1
        offsets.append(offset)
        if iter_preop == maxIter_preop or np.max(np.abs(offset)) < minDiff_preop: break
    
    print(f'Pre-optimization ended after {iter_preop} iterations.')
    print(f'Added offsets: {np.sum(np.stack(offsets), axis=0)}')
    
    # Outlier rejection
    stdev = np.std(displacements[foundTags])
    outliers = np.any(np.abs(displacements)>outlierThreshold*stdev, axis=(3,4))
    foundTags[outliers] = False
    tagWorldPoints[~foundTags] = np.nan
    print(f'Frames per tag after outlier rejection: {np.sum(foundTags, axis=(0,2))}')
    
    reportError(tagBoardPoints, cameraMatrices, tagImgPoints, tagWorldPoints, foundTags, charucoOrigins if TARGET else None)
    
    # Jacobian sparsity matrix
    foundCams = np.any(foundTags, axis=1)
    residualsPerTag = 12 if MINPOS else 8  # 4 points per tag and x,y,(z) residuals for each point
    # if TARGET: N_residuals = np.sum(foundTags) * residualsPerTag + np.sum(foundCams) * 2
    N_residuals = np.sum(foundTags) * residualsPerTag
    N_params = tagBoardPoints.size + (N_cams*3 if TARGET else 0)
    jac_sparsity = lil_matrix((N_residuals, N_params), dtype=int)
    if TARGET: jac_sparsity[:,-N_cams*3:] = 1
    nextResidual = 0
    for frame, camIndex in zip(*np.where(foundCams)):
        tags_frame = np.where(foundTags[frame, :, camIndex])[0]
        N_residuals_frame = np.sum(foundTags[frame, :, camIndex]) * residualsPerTag
        for tag in tags_frame:
            jac_sparsity[nextResidual:nextResidual+N_residuals_frame, tag*12:tag*12+12] = 1
        nextResidual += residualsPerTag
    # if TARGET: jac_sparsity[nextResidual:,:] = 1
    
    # Initial params
    # if TAGVECTORS:
    #     tagVectors = np.full((N_tags,2,3), np.nan, dtype=np.float32)  # first index at each tag is a rotation vector, second is a translation vector 
    #     for tag in range(N_tags):
    #         R, T = rigidTransform(tagPoints.T, tagObjPoints[tag].T)
    #         tagVectors[tag,0] = cv2.Rodrigues(R)[0].T
    #         tagVectors[tag,1] = T.T
    #     params_initial = tagVectors.ravel()
    if TARGET:
        params_initial = np.hstack([tagBoardPoints.ravel(), charucoOrigins.ravel()])
    else:
        params_initial = tagBoardPoints.ravel()
    

# %%
    # Optimization
    if MINPOS: residuals_initial = computeResiduals_disp(params_initial, tagWorldPoints, foundTags)
    else: residuals_initial = computeResiduals_proj(params_initial, cameraMatrices, tagImgPoints, tagWorldPoints, foundTags)
    rmse_initial = np.sqrt(np.mean(residuals_initial**2))
    print(f'Initial RMSE of residuals: {rmse_initial}')
    print('Starting bundle adjustment.')
    if MINPOS: lsqResult = least_squares(computeResiduals_disp, params_initial, max_nfev=maxIter_lsq, diff_step=.001, jac_sparsity=jac_sparsity, verbose=2, x_scale='jac', method='trf', loss='linear', args=(tagWorldPoints, foundTags))
    else: lsqResult = least_squares(computeResiduals_proj, params_initial, max_nfev=maxIter_lsq, diff_step=.001, jac_sparsity=jac_sparsity, verbose=2, x_scale='jac', method='trf', loss='linear', args=(cameraMatrices, tagImgPoints, tagWorldPoints, foundTags))
    residuals_final = lsqResult.fun
    rmse_final = np.sqrt(np.mean(residuals_final**2))
    print(f'Final RMSE of residuals: {rmse_final}')
    
    # if TAGVECTORS:
    #     tagVectors_final = lsqResult.x.reshape(N_tags, 2, 3)
    #     tagObjPoints_final = np.full((N_tags, 4, 3), np.nan, dtype=np.float32)
    #     for tag in range(N_tags):
    #         rvec, tvec = tagVectors[tag]
    #         R = cv2.Rodrigues(rvec)[0]
    #         tagObjPoints_final[tag] = tagPoints @ R.T + tvec
    if TARGET:
        tagBoardPoints_final = lsqResult.x[:-3*N_cams].reshape(N_tags, 4, 3)
        tipPositions = lsqResult.x[-3*N_cams:].reshape(N_cams,3,1)
    else:
        tagBoardPoints_final = lsqResult.x.reshape(N_tags, 4, 3)
        
    reportError(tagBoardPoints_final, cameraMatrices, tagImgPoints, tagWorldPoints, foundTags, tipPositions if TARGET else None)
    
# %%
    fileName = f'target_{time.strftime("%Y-%m-%d_%H_%M_%S")}' if TARGET else f'ref_{time.strftime("%Y-%m-%d_%H_%M_%S")}'
    np.savetxt(f'calib_files/tool/{fileName}.txt', tagBoardPoints_final.reshape(N_tags, 12))
# %%
