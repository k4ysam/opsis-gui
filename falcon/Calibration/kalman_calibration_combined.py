import numpy as np
import os
import json
import matplotlib.pyplot as plt

# Flags to control whether to include average mono and stereo data in plots
INCLUDE_MONO_AVERAGE = False  # Set to False to omit the mono average column
INCLUDE_STEREO = False        # Set to False to omit the stereo column

folder_path = '../Registration/Landmark_Registration_Trials/GT01_init/_GT01_LEFT'  # Set the folder containing JSON files

# Thresholds for filtering out outliers in pose data (adjust based on your data range)
POSITION_THRESHOLD = 10000  # Threshold for position elements (x, y, z)
ROTATION_THRESHOLD = 10000  # Threshold for rotation elements (yaw, pitch, roll)

# Flags
MONO_STEREO_FLAG = 'mono'  # Options: 'mono' or 'stereo'
XYZ_FLAG = False  # If True, compute norms using only XYZ elements (first 3 pose elements)

# Global variables to hold data
covariances = {cam: [] for cam in range(5)}  # Assuming 5 cameras
norms = {cam: [] for cam in range(5)}
stereo_covariances = []
stereo_norms = []
positions = []  # Store position names
mean_covariances = {cam: [] for cam in range(5)}
mean_norms = []

# Create folder for saving figures
save_folder = 'noise_figs'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ---------------------------
# Pose-Level Filtering & Covariance Computation
# ---------------------------
def filter_outliers_percentile_poses(poses, lower_percentile=1, upper_percentile=85):
    """
    Filters outliers dimension-by-dimension based on percentiles.
    :param poses: list of pose vectors, each of length 6 [x, y, z, yaw, pitch, roll]
    :param lower_percentile: the lower percentile cutoff (e.g., 1)
    :param upper_percentile: the upper percentile cutoff (e.g., 99)
    :return: filtered list of poses
    """
    if not poses:
        return poses
    
    arr = np.array(poses)
    # Compute the lower and upper percentile for each of the 6 columns
    lower_bounds = np.percentile(arr, lower_percentile, axis=0)  
    upper_bounds = np.percentile(arr, upper_percentile, axis=0)

    # Keep only rows (poses) that lie within [lower_bound, upper_bound] for *all* 6 dims
    mask = np.all((arr >= lower_bounds) & (arr <= upper_bounds), axis=1)
    filtered_arr = arr[mask]
    return filtered_arr.tolist()

def detect_outliers(poses, k=3):
    """
    Detect and remove outlier poses using the Median Absolute Deviation (MAD) method.
    :param poses: List of poses (each pose is a list of 6 elements: [x, y, z, yaw, pitch, roll]).
    :param k: Threshold for MAD (default is 3).
    :return: List of filtered poses.
    """
    if not poses:
        return poses

    poses_array = np.array(poses)
    median = np.median(poses_array, axis=0)
    mad = np.median(np.abs(poses_array - median), axis=0)
    mad[mad == 0] = 1e-6  # Avoid division by zero
    modified_z_scores = 0.6745 * (poses_array - median) / mad
    valid_indices = np.all(np.abs(modified_z_scores) < k, axis=1)
    filtered_poses = poses_array[valid_indices]
    return filtered_poses.tolist()

def filter_outliers(poses):
    """Keep only poses whose position and rotation are within preset thresholds."""
    filtered_poses = []
    for pose in poses:
        position = pose[:3]
        rotation = pose[3:]
        if all(abs(p) < POSITION_THRESHOLD for p in position) and all(abs(r) < ROTATION_THRESHOLD for r in rotation):
            filtered_poses.append(pose)
    return filtered_poses

def compute_R(pose_data):
    """Compute a diagonal covariance matrix (variances along the diagonal) from pose data."""
    variances = np.var(pose_data, axis=0, dtype=np.float64)
    R = np.diag(variances)
    return R

def parse_and_compute_R(json_file, position_name):
    with open(json_file, 'r') as f:
        data = json.load(f)
    positions.append(position_name)
    # Process mono data for each camera
    for cam in range(5):
        cam_poses = []
        for frame_poses in data["5 Cam Mono"]["poses"]:
            cam_poses.append(frame_poses[str(cam)])
        
        # Optionally apply filtering methods
        # cam_poses = filter_outliers(cam_poses)
        filtered_poses = filter_outliers_percentile_poses(cam_poses, lower_percentile=1, upper_percentile=99)
        if filtered_poses:
            R = compute_R(filtered_poses)
            covariances[cam].append(R)
            norm = np.linalg.norm(R[:3, :3], 'fro') if XYZ_FLAG else np.linalg.norm(R, 'fro')
            norms[cam].append(norm)

    # Process stereo data if available
    if "5 Cam Stereo" in data and "poses" in data["5 Cam Stereo"]:
        stereo_poses = data["5 Cam Stereo"]["poses"]
        # stereo_poses = filter_outliers(stereo_poses)
        filtered_poses = filter_outliers_percentile_poses(stereo_poses, lower_percentile=1, upper_percentile=99)
        if filtered_poses:
            R_stereo = compute_R(filtered_poses)
            stereo_covariances.append(R_stereo)
            norm_stereo = np.linalg.norm(R_stereo[:3, :3], 'fro') if XYZ_FLAG else np.linalg.norm(R_stereo, 'fro')
            stereo_norms.append(norm_stereo)

# ---------------------------
# MAD-Based Filtering at the Position Level
# ---------------------------
def filter_norms_by_position_mad(positions, norms, stereo_norms, k=3):
    """
    Filters out positions based on the average mono norm across cameras.
    :param positions: List of position names.
    :param norms: Dictionary of norms for each camera.
    :param stereo_norms: List of stereo norms.
    :param k: MAD threshold multiplier.
    :return: Filtered positions, norms, and stereo_norms.
    """
    mono_norm_matrix = np.array([norms[cam] for cam in range(5)])
    avg_norm_per_position = np.mean(mono_norm_matrix, axis=0)
    median_global = np.median(avg_norm_per_position)
    mad = np.median(np.abs(avg_norm_per_position - median_global))
    if mad == 0:
        return positions, norms, stereo_norms
    valid_idx = np.where(np.abs(avg_norm_per_position - median_global) <= k * mad)[0]
    positions_filtered = [positions[i] for i in valid_idx]
    for cam in range(5):
        norms[cam] = [norms[cam][i] for i in valid_idx]
    if stereo_norms:
        stereo_norms = [stereo_norms[i] for i in valid_idx]
    return positions_filtered, norms, stereo_norms

# ---------------------------
# Mean Covariance Computation
# ---------------------------
def compute_R_mean():
    mean_norms = []
    if MONO_STEREO_FLAG == 'mono':
        for cam in range(5):
            if covariances[cam]:
                stacked_covariances = np.stack(covariances[cam])
                R_mean = np.mean(stacked_covariances, axis=0)
                mean_covariances[cam].append(R_mean)
                R_mean_norm = np.linalg.norm(R_mean[:3, :3], 'fro') if XYZ_FLAG else np.linalg.norm(R_mean, 'fro')
                mean_norms.append(R_mean_norm)
                print(f"Camera {cam+1}: Mean R Norm = {R_mean_norm}")
    if stereo_norms:
        stereo_mean_norm = np.mean(stereo_norms)
        mean_norms.append(stereo_mean_norm)
        print(f"Stereo: Mean R Norm Across All Positions = {stereo_mean_norm}")
    return mean_norms

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_r_norm_matrix():
    # Prepare data for heatmap: mono norms from 5 cameras (rows: positions, columns: cameras)
    mono_norm_matrix = np.array([norms[cam] for cam in range(5)]).T
    columns = []  # will hold each column for the heatmap
    camera_labels = []
    for cam in range(5):
        columns.append(mono_norm_matrix[:, cam])
        camera_labels.append(f'Cam {cam+1}')
    if INCLUDE_MONO_AVERAGE:
        mono_avg_norms = np.mean(mono_norm_matrix, axis=1)
        columns.append(mono_avg_norms)
        camera_labels.append('Mono\nAverage')
    if INCLUDE_STEREO and stereo_norms:
        stereo_norm_column = np.array(stereo_norms)
        columns.append(stereo_norm_column)
        camera_labels.append('Stereo')
    r_norm_matrix = np.column_stack(columns)
    num_cameras = r_norm_matrix.shape[1]
    num_positions = len(positions)
    fig_width = num_cameras * 1.2
    fig_height = num_positions * 0.4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.imshow(r_norm_matrix, cmap='viridis', aspect='auto')
    fig.colorbar(cax)
    # plt.title('Frobenius Norms of R Matrices for Each Sampled Position')
    plt.ylabel('Position')
    plt.xticks(np.arange(num_cameras), camera_labels)
    plt.yticks(np.arange(len(positions)), positions)
    data_min, data_max = np.min(r_norm_matrix), np.max(r_norm_matrix)
    range_val = data_max - data_min
    text_threshold = np.median(r_norm_matrix) + 0.5 * range_val
    for i in range(r_norm_matrix.shape[0]):
        for j in range(r_norm_matrix.shape[1]):
            text_color = 'black' if r_norm_matrix[i, j] > text_threshold else 'white'
            plt.text(j, i, f'{r_norm_matrix[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{save_folder}/r_norm_matrix_mono_stereo.png')
    plt.show()

def plot_violin_norms():
    plt.figure()
    data = [norms[cam] for cam in range(5)]
    labels = [f'Cam {i+1}' for i in range(5)]
    if INCLUDE_MONO_AVERAGE:
        mono_avg_norms = np.mean([norms[cam] for cam in range(5)], axis=0)
        data.append(mono_avg_norms)
        labels.append('Mono\nAverage')
    if INCLUDE_STEREO and stereo_norms:
        data.append(stereo_norms)
        labels.append('Stereo')
    parts = plt.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_edgecolor('black')
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.title('Distribution of R Matrix Frobenius Norms Across Sampled Positions')
    plt.ylabel('Norm [mm]')
    plt.grid(False)
    plt.savefig(f'{save_folder}/violin_norms_mono_stereo.png')
    plt.show()

def plot_boxplot_norms():
    plt.figure()
    data = []
    labels = []
    for cam in range(5):
        data.append(norms[cam])
        labels.append(f'Cam {cam+1}')
    if INCLUDE_MONO_AVERAGE:
        lengths = [len(norms[cam]) for cam in range(5)]
        if len(set(lengths)) == 1 and lengths[0] > 0:
            mono_avg = np.mean(np.array([norms[cam] for cam in range(5)]), axis=0)
            data.append(mono_avg)
            labels.append('Mono\nAverage')
    if INCLUDE_STEREO and stereo_norms:
        data.append(stereo_norms)
        labels.append('Stereo')
    if not data:
        print("No data to plot in boxplot.")
        return
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.title('Distribution of R Matrix Frobenius Norms Across Sampled Positions')
    plt.ylabel('Norm [mm]')
    plt.grid(False)
    plot_file = f'{save_folder}/boxplot_norms_mono_stereo.png'
    plt.savefig(plot_file)
    plt.show()

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Process each JSON file in the folder
    for json_file in sorted(os.listdir(folder_path)):
        if json_file.endswith('.json'):
            position_name = json_file.split('_')[1]
            print(f"Processing {json_file} for position {position_name}")
            parse_and_compute_R(os.path.join(folder_path, json_file), position_name)
    
    # Optionally, apply MAD-based filtering to remove outlier positions
    # positions, norms, stereo_norms = filter_norms_by_position_mad(positions, norms, stereo_norms, k=1)
    
    # Compute and print mean covariance norms
    mean_norms = compute_R_mean()
    
    # Generate plots
    plot_r_norm_matrix()
    plot_violin_norms()
    plot_boxplot_norms()