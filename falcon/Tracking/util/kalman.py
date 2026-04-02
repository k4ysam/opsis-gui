from collections import deque
import numpy as np

class KalmanFilterCV():
    """Constant Velocity Kalman Filter."""
    def __init__(self, freq=60, q_path='../Calibration/calib_files/filter/Q_matrix_mono.txt', calib=True):
        """
        Initializes the Kalman Filter with given parameters.

        Args:
            freq (float): Frequency of measurements in Hz (default 60).
            process_noise (float): Noise of dynamics (process) model (default = 1mm)
        """
        # States = x, y, z, yaw, pitch, roll, dx, dy, dz, dyaw, dpitch, droll
        # []
        dt = 1 / freq
        # x_k = A * x_k-1
        self.A = np.array([[1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        self.B = 0
        # Measurement is only x, not dx
        # y = Cx
        self.C = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        
        # Queue of last 30 poses, used to determine process noise matrix R
        self.pose_queue = deque(maxlen=30)

        if calib:
            # Load the process noise matrix Q from .txt file
            Q_loaded = np.loadtxt(q_path, delimiter=',')  # Load the Q matrix from the .txt file
            self.Q = np.zeros((12, 12))
            self.Q[:6, :6] = Q_loaded

            # Load the measurement noise matrix R from .txt file
            self.R = np.loadtxt(q_path, delimiter=',')  # Initial R is just Q

        else:
            self.Q = np.eye(12,12) * 0.001
            self.R = np.diag([0.008,0.008,0.01,0.0008,0.0008,0.0008])

        
        self.P_k = np.zeros((12,12))
        self.K_k = None
        # State
        self.x = None
        # Measurements
        self.u_acc = None
        self.y_k = None
        self.iniated_flag = False

    def initiate_state(self, x0):
        """
        Initializes the state of the Kalman filter.

        Args:
            x0 (np.ndarray): Initial state vector. Shape must be (6,1).
        """
        # State
        self.x = np.vstack((x0,np.zeros((6,1))))
        self.iniated_flag = True
    
    def has_been_initiated(self):
        return self.iniated_flag

    def set_measurement(self, y_k):
        """
        Sets the measurement for the Kalman filter.

        Args:
            y_k (np.ndarray): Measurement vector. Measurement must be (6,1).
        """
        self.u_acc = 0
        self.y_k = y_k

    def predict(self) -> np.ndarray:
        """
        Prediction step of the Kalman filter.

        Returns:
            np.ndarray: Predicted state vector, of shape (12,1). First 6 contain pose.
        """
        # Prediction step
        self.x = self.A @ self.x 
        self.P_k = self.A @ self.P_k @ self.A.T + self.Q
        return self.x
    
    def correct(self) -> None:
        """
        Correction step of the Kalman filter.
        """
        # Correction Step
        tmp = (self.C @ self.P_k @ self.C.T) + self.R
        tmp += np.eye(tmp.shape[0]) * 1e-5 # Regularization
        self.K_k = self.P_k @ self.C.T @ np.linalg.solve(tmp,np.eye(tmp.shape[0],tmp.shape[1]))
        self.x = self.x + self.K_k @ (self.y_k - (self.C @ self.x))
        
        self.P_k = (np.eye(self.K_k.shape[0],self.C.shape[1]) - 
                    (self.K_k @ self.C)) @ self.P_k
        
        return self.x
    def set_dt(self,dt: float) -> None:
        """
        Sets the time interval between measurements.

        Args:
            dt (float): Time interval between measurements.
        """
        self.A = np.array([[1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dt],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
    def set_measurement_matrices(self, num_measurements: int, new_R: np.ndarray)-> None: 
        """
        Sets measurement matrix (C) and measurement covariance matrix (R) based upon number of cameras 
        that have published data.
        Args: 
        num_measurements (int): Number of measurements/cameras that returned a pose estimate
        new_R (np.ndarray): Associated covariance matrix. For each measurement, covariance should be a
                            diagonal 6x6 matrix.  
        """
        one_measurement = np.block([[np.eye(6),np.zeros((6,6))]])
        self.C = np.block([[np.eye(6),np.zeros((6,6))]])
        for _ in range(num_measurements-1):
            self.C = np.vstack((self.C,one_measurement))

        self.R = new_R
        return
    
    def compute_measurement_noise(self, poses: list) -> None:
        """
        Computes measurement covariance matrix (R) based upon number of cameras 
        that have published data.
        Args: 
        poses (list): List of poses last obtained by each camera.
        """
        # Add latest poses to queue
        self.pose_queue.append(poses)

        # Initialize list to store R for each camera
        covars = []

        # Initialize a list of lists to store poses for each camera
        num_cameras = len(poses)
        camera_poses = [[] for _ in range(num_cameras)]

        # Organize poses by camera
        for pose_set in self.pose_queue:
            for cam_id, pose in enumerate(pose_set):
                if pose is not None:  # Only add non-None poses
                    camera_poses[cam_id].append(pose.flatten())

        # Compute covariance matrices for each camera only when 30 poses are collected
        for cam_id, poses in enumerate(camera_poses):
            if poses:  # Check if there are any poses to concatenate
                all_poses = np.vstack(poses)
                variance = np.var(all_poses, axis=0, dtype=np.float64)
                R = np.diag(variance.flatten())
                covars.append(R)
                # Compute and store the norm of the R matrix
                norm_R = np.linalg.norm(R, 'fro')
                norm_R_xyz = np.linalg.norm(R[:3, :3], 'fro')
                print(f"Camera {cam_id+1}: Norm of R = {norm_R:.4f}; XYZ Norm = {norm_R_xyz:.4f}")
            else:
                # Append a zero matrix if the camera cannot see the target
                R = np.zeros((6, 6))
                covars.append(R)
                print(f"Camera {cam_id+1}: Cannot see the target.")

        return covars
    
    def set_process_noise(self, covars: np.ndarray):
        """
        Computes process covariance matrix (Q) based upon number of cameras 
        that have published data.
        Args: 
        covars (np.ndarray): Numpy array of measurement covariance matrices R last computed for each camera.
        """
        if len(covars) == 0:
            print("No covariance matrices available to set process noise.")
            return

        # Stack the R's
        stacked_covars = np.stack(covars, axis=0)

        # Calculate the mean across all R's
        mean_R = np.mean(stacked_covars, axis=0)
        new_Q = np.zeros((12, 12))
        new_Q[:6, :6] = mean_R
        self.Q = new_Q