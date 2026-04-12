import numpy as np
import os
import json
import matplotlib.pyplot as plt

folder_path = '../Registration/Landmark_Registration_Trials/GT01_init/_GT01_TOP'  # Set the folder containing JSON files

# Thresholds for filtering out outliers (adjust based on your data range)
POSITION_THRESHOLD = 10000  # Threshold for position elements (x, y, z)
ROTATION_THRESHOLD = 10000  # Threshold for rotation elements (yaw, pitch, roll)
# POSITION_THRESHOLD = 250  # Threshold for position elements (x, y, z)
# ROTATION_THRESHOLD = 200  # Threshold for rotation elements (yaw, pitch, roll)

# Flags
MONO_STEREO_FLAG = 'mono'  # Options: 'mono' or 'stereo'
XYZ_FLAG = False  # If True, compute norms using only XYZ elements (first 3 pose elements)

# Global variables to hold data
covariances = {cam: [] for cam in range(5)}  # Assuming 5 cameras
norms = {cam: [] for cam in range(5)}
mean_covariances = {cam: [] for cam in range(5)}
mean_norms = []
stereo_covariances = []
stereo_norms = []
positions = []  # Store position names
Q_matrix = None  # To store the final Q matrix
R_30x30 = None  # Store 30x30 R matrix for mono

# Create folder for saving figures
save_folder = 'noise_figs'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def filter_outliers_percentile_poses(poses, lower_percentile=1, upper_percentile=99):
    """
    Filters outliers dimension-by-dimension based on percentiles.

    :param poses: list of pose vectors, each of length 6 [x, y, z, yaw, pitch, roll]
    :param lower_percentile: the lower percentile cutoff (e.g. 1)
    :param upper_percentile: the upper percentile cutoff (e.g. 99)
    :return: filtered list of poses
    """
    if not poses:
        return poses
    
    arr = np.array(poses)  # shape: (N, 6)
    # Compute the lower and upper percentile for each of the 6 columns
    lower_bounds = np.percentile(arr, lower_percentile, axis=0)  # shape: (6,)
    upper_bounds = np.percentile(arr, upper_percentile, axis=0)  # shape: (6,)

    # Keep only rows (poses) that lie within [lower_bound, upper_bound] for *all* 6 dims
    mask = np.all((arr >= lower_bounds) & (arr <= upper_bounds), axis=1)
    filtered_arr = arr[mask]
    return filtered_arr.tolist()

# Function to filter out outliers
def filter_outliers(poses):
    filtered_poses = []
    for pose in poses:
        position = pose[:3]
        rotation = pose[3:]
        if all(abs(p) < POSITION_THRESHOLD for p in position) and all(abs(r) < ROTATION_THRESHOLD for r in rotation):
            filtered_poses.append(pose)
    return filtered_poses

# Function to compute the covariance matrix R
def compute_R(pose_data):
    elements = np.array(pose_data).T
    if XYZ_FLAG:
        elements = elements[:3, :]  # Use only XYZ
    variances = np.var(pose_data, axis=0, dtype=np.float64)
    R = np.diag(variances)
    return R

# Function to parse a JSON file and compute R matrices
def parse_and_compute_R(json_file, position_name):
    with open(json_file, 'r') as f:
        data = json.load(f)

    positions.append(position_name)

    if MONO_STEREO_FLAG == 'mono':
        for cam in range(5):  # Assuming 5 cameras
            cam_poses = []
            for frame_poses in data["5 Cam Mono"]["poses"]:
                cam_poses.append(frame_poses[str(cam)])
            
            # 1) (Optional) Filter using absolute thresholds
            cam_poses = filter_outliers(cam_poses)
            # 2) Filter dimension-by-dimension based on percentile
            cam_poses = filter_outliers_percentile_poses(cam_poses, lower_percentile=1, upper_percentile=80)
            
            if cam_poses:
                R = compute_R(cam_poses)
                covariances[cam].append(R)
                norm_val = np.linalg.norm(R, 'fro')
                norms[cam].append(norm_val)

    elif MONO_STEREO_FLAG == 'stereo':
        stereo_poses = data["5 Cam Stereo"]["poses"]
        
        # 1) (Optional) Filter using absolute thresholds
        stereo_poses = filter_outliers(stereo_poses)
        # 2) Filter dimension-by-dimension based on percentile
        stereo_poses = filter_outliers_percentile_poses(stereo_poses, lower_percentile=1, upper_percentile=99)
        
        if stereo_poses:
            R_stereo = compute_R(stereo_poses)
            stereo_covariances.append(R_stereo)
            norm_stereo = np.linalg.norm(R_stereo, 'fro')
            stereo_norms.append(norm_stereo)

# Function to compute the mean covariance matrix R for each camera
def compute_R_mean():
    if MONO_STEREO_FLAG == 'mono':
        for cam in range(5):
            if covariances[cam]:
                stacked_covariances = np.stack(covariances[cam])
                R_mean = np.mean(stacked_covariances, axis=0)
                mean_covariances[cam].append(R_mean)
                R_mean_norm = np.linalg.norm(R_mean, 'fro')
                mean_norms.append(R_mean_norm)
                print(f"Camera {cam+1}: Mean R Norm = {R_mean_norm}")
    elif MONO_STEREO_FLAG == 'stereo':
        stacked_covariances = np.stack(stereo_covariances)
        R_mean = np.mean(stacked_covariances, axis=0)
        mean_covariances[0].append(R_mean)
        R_mean_norm = np.mean([np.linalg.norm(R, 'fro') for R in stereo_covariances])
        mean_norms.append(R_mean_norm)
        print(f"Stereo: Mean R Norm Across All Positions = {R_mean_norm}")

# Function to compute the overall mean covariance matrix Q and the 30x30 R matrix
def compute_Q_and_R_30x30():
    global Q_matrix, R_30x30
    if MONO_STEREO_FLAG == 'mono':
        all_mean_covariances = np.stack([mean_covariances[cam][0] for cam in range(5)])
        Q = np.mean(all_mean_covariances, axis=0)
        
        # Create the 30x30 matrix from the 5x6x6 mean covariances
        R_30x30 = np.zeros((30, 30))
        for i, R_mean in enumerate(all_mean_covariances):
            start_idx = i * 6
            end_idx = start_idx + 6
            R_30x30[start_idx:end_idx, start_idx:end_idx] = R_mean

        print(f"Overall mean covariance matrix Q:\n{Q}")
        print(f"30x30 R matrix:\n{R_30x30}")
    elif MONO_STEREO_FLAG == 'stereo':
        Q = np.mean(mean_covariances[0], axis=0)
    Q_matrix = Q

# Function to save the Q and R_30x30 matrix as .txt
def save_matrices():
    if Q_matrix is not None:
        np.savetxt(f"calib_files/filter/Q_matrix_{MONO_STEREO_FLAG}.txt", Q_matrix, delimiter=',', fmt='%.8f')
        print(f"Q matrix saved to calib_files/filter/Q_matrix_{MONO_STEREO_FLAG}.txt")
    
    if R_30x30 is not None:
        np.savetxt(f"calib_files/filter/R_30x30_mono.txt", R_30x30, delimiter=',', fmt='%.8f')
        print(f"30x30 R matrix saved to calib_files/filter/R_30x30_mono.txt")

def filter_outliers_percentile(data, lower_percentile=1, upper_percentile=75):
    """
    Filters outliers based on percentiles.
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return [x for x in data if lower_bound <= x <= upper_bound]

# Plotting function for camera norms using violin plots
def plot_cam_norms_violin():
    plt.figure()
    if MONO_STEREO_FLAG == 'mono':
        # Prepare data for violin plot
        data = [norms[cam] for cam in range(5)]  # Filter outliers
        labels = [f'{cam+1}' for cam in range(5)]
        
        # Create violin plot
        plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks(range(1, 6), labels)  # Label x-axis with camera numbers
        plt.title('Distribution of Frobenius Norms for Each Camera (Mono)')
        file_name = f'{save_folder}/violin_norms_mono.png'
    elif MONO_STEREO_FLAG == 'stereo':
        # Create violin plot for stereo
        # filtered_stereo_norms = filter_outliers_percentile(stereo_norms)  # Filter outliers
        plt.violinplot([stereo_norms], showmeans=True, showmedians=True)
        plt.xticks([1], ['Stereo'])  # Label x-axis with "Stereo"
        plt.title('Distribution of Frobenius Norms for Stereo')
        file_name = f'{save_folder}/violin_norms_stereo.png'
    
    plt.xlabel('Camera' if MONO_STEREO_FLAG == 'mono' else 'Mode')
    plt.ylabel('Norm [mm]')
    plt.grid(False)
    plt.savefig(file_name)  # Save the figure in noise_figs
    plt.show()

def plot_cam_norms_boxplot():
    plt.figure()
    if MONO_STEREO_FLAG == 'mono':
        # Prepare data for box plot
        data = [norms[cam] for cam in range(5)]
        labels = [f'{cam+1}' for cam in range(5)]
        
        # Create box plot
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.title('Distribution of Frobenius Norms for Each Camera (Mono)')
        file_name = f'{save_folder}/boxplot_norms_mono.png'
    elif MONO_STEREO_FLAG == 'stereo':
        # Create box plot for stereo
        plt.boxplot([stereo_norms], labels=['Stereo'], showmeans=True)
        plt.title('Distribution of Frobenius Norms for Stereo')
        file_name = f'{save_folder}/boxplot_norms_stereo.png'
    
    plt.xlabel('Camera' if MONO_STEREO_FLAG == 'mono' else 'Mode')
    plt.ylabel('Norm [mm]')
    plt.grid(False)
    plt.savefig(file_name)  # Save the figure in noise_figs
    plt.show()

# Plotting function for R norm matrix and saving
def plot_r_norm_matrix():
    if MONO_STEREO_FLAG == 'mono':
        r_norm_matrix = np.array([norms[cam] for cam in range(5)]).T
        
        num_positions = len(positions)
        num_cameras = 5
        fig_width = num_cameras * 1.2
        fig_height = num_positions * 0.4
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        cax = ax.imshow(r_norm_matrix, cmap='viridis', aspect='auto')
        fig.colorbar(cax)

        plt.title('Frobenius Norms of R Matrices for Each Camera and Position (Mono)')
        plt.xlabel('Camera')
        plt.ylabel('Position')

        plt.xticks(np.arange(num_cameras), [f'{cam+1}' for cam in range(num_cameras)])
        sorted_positions = sorted(positions)
        plt.yticks(np.arange(len(sorted_positions)), sorted_positions)

        for i in range(len(sorted_positions)):
            for j in range(num_cameras):
                plt.text(j, i, f'{r_norm_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{save_folder}/r_norm_matrix_mono.png')  # Save the figure in noise_figs
        plt.show()
    elif MONO_STEREO_FLAG == 'stereo':
        plt.figure()
        sorted_positions = sorted(positions)  # Ensure the positions are sorted
        sorted_stereo_norms = [stereo_norms[positions.index(pos)] for pos in sorted_positions]

        bars = plt.bar(range(len(sorted_positions)), sorted_stereo_norms)
        plt.title('Frobenius Norms of R Matrices for Stereo at Each Position')
        plt.xlabel('Position')
        plt.ylabel('Stereo Norm [mm]')
        plt.xticks(range(len(sorted_positions)), sorted_positions)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'{save_folder}/r_norm_matrix_stereo.png')  # Save the figure in noise_figs
        plt.show()

if __name__ == "__main__":
    for json_file in os.listdir(folder_path):
        if json_file.endswith('.json'):
            position_name = json_file.split('_')[1]
            print(f"Processing {json_file} for position {position_name}")
            parse_and_compute_R(os.path.join(folder_path, json_file), position_name)

    compute_R_mean()
    compute_Q_and_R_30x30()

    # Save Q matrix and 30x30 R matrix to files
    save_matrices()

    # Plot and save results
    plot_cam_norms_boxplot()
    plot_cam_norms_violin()
    plot_r_norm_matrix()