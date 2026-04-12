import numpy as np

fixed_points = np.array([
    [77.24, -30.5544, 21.0519],
    [39.5306, -9.2923, 85.0889],
    [-14.3043, -72.0667, 72.605],
    [19.3012, -88.9711, 3.9566],
    [31.4759, -11.4459, -66.5619],
    [77.5623, -19.2732, -3.6433],
    [40.9331, -80.6808, 0.0667],
    [5.6701, -80.7094, -36.2677]
])

moving_points = np.array([
    [-182.493, 45.6969, 45.595],
    [-242.955, 92.5608, 20.9376],
    [-286.781, 58.8361, 82.8754],
    [-230.68, 8.4968, 107.225],
    [-191.635, -52.3737, 27.5795],
    [-172.093, 21.6516, 33.6549],
    [-208.232, 13.3397, 96.1123],
    [-227.358, -35.4241, 98.4835]
])

def calculate_TRE(fixed=None, moving=None, R=None, T=None, tag_file=None, xfm_file=None):
    """
    Calculates the target registration error (TRE) by applying the transformation to
    the moving points and comparing them to the fixed points. Points can be read from
    a .tag file and transformation can be read from a .xfm file.

    :param tag_file: Path to the .tag file for fixed and moving points (optional)
    :param xfm_file: Path to the .xfm file for R and T (optional)
    :param fixed: Fixed point set, N x 3 ndarray (optional)
    :param moving: Moving point set, N x 3 ndarray (optional)
    :param R: Rotation matrix, 3x3 ndarray (optional)
    :param T: Translation vector, 3x1 ndarray (optional)
    :return: FTE for each point, overall RMSE
    """
    # Read points from .tag file if provided
    if tag_file:
        fixed, moving = read_tag_file(tag_file)

    # Read R and T from .xfm file if provided
    if xfm_file:
        R, T = read_xfm_file(xfm_file)

    if fixed is None or moving is None:
        raise ValueError("Either fixed and moving points must be provided, or a tag_file must be specified.")

    if R is None or T is None:
        raise ValueError("Either R and T must be provided, or xfm_file must be specified.")

    # Transform the moving points
    transformed_moving = apply_transform(moving, R, T)

    # Calculate per point error
    ppe = []
    per_point_error = np.linalg.norm(fixed - transformed_moving, axis=1)

    # Print TRE for each point
    print("\nTRE for each point:")
    for i, tre in enumerate(per_point_error):
        ppe.append(tre)
        print(f"Point {i}: {tre:.6f} mm")

    # Compute RMSE
    rmse = np.sqrt(np.mean(per_point_error**2))
    print(f"\nRMSE: {rmse:.6f} mm")

    return per_point_error, rmse, ppe


def read_xfm_file(filename):
    """
    Reads the rotation matrix and translation vector from a .xfm file.

    :param filename: Path to the .xfm file
    :return: Rotation matrix (R), translation vector (T)
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Find the line starting with "Linear_Transform ="
        for line in lines:
            if line.startswith('Linear_Transform'):
                # Extract the R and T from the subsequent lines
                r_t_lines = lines[lines.index(line) + 1:lines.index(line) + 4]
                rt_matrix = np.array([[float(val) for val in l.strip().split()] for l in r_t_lines])
                R = rt_matrix[:, :3]
                T = rt_matrix[:, 3]
                return R, T
    raise ValueError(f"Invalid .xfm file format: {filename}")

def apply_transform(points, R=None, T=None, xfm_file=None):
    """
    Applies the transformation to the given points.

    :param points: Point set, N x 3 ndarray
    :param R: Rotation matrix, 3x3 ndarray (optional)
    :param T: Translation vector, 3x1 ndarray (optional)
    :param xfm_file: Path to the .xfm file (optional)
    :return: Transformed point set, N x 3 ndarray
    """
    if xfm_file:
        R, T = read_xfm_file(xfm_file)
    
    if R is None or T is None:
        raise ValueError("Either R and T must be provided, or xfm_file must be specified.")
    
    transformed_points = np.dot(points, R.T) + T
    return transformed_points

def read_tag_file(filename):
    """
    Reads fixed and moving points from a .tag file.

    :param filename: Path to the .tag file
    :return: Fixed and moving point sets, each as an N x 3 ndarray
    """
    fixed_points = []
    moving_points = []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip lines that do not contain point data
            if line.startswith('Points =') or not line or line.startswith('%') or line.startswith('Volumes'):
                continue
            
            parts = line.split()
            try:
                fixed_point = list(map(float, parts[:3]))
                moving_point = list(map(float, parts[3:6]))
                fixed_points.append(fixed_point)
                moving_points.append(moving_point)
            except ValueError:
                # Log or print an error message for debugging, if needed
                print(f"Skipping invalid line: {line}")
                continue

    return np.array(fixed_points), np.array(moving_points)

def register(fixed=None, moving=None, filename='transformation.xfm', tag_file=None):
    """
    Performs point-based registration using the Orthogonal Procrustes method,
    prints the rotation matrix, translation vector, and FRE, saves the 
    transformation to a .xfm file, and prints the FRE for each point.

    :param fixed: Point set, N x 3 ndarray (optional)
    :param moving: Point set, N x 3 ndarray of corresponding points (optional)
    :param filename: Filename for saving the transformation
    :param tag_file: Path to the .tag file (optional)
    """
    if tag_file:
        fixed, moving = read_tag_file(tag_file)
    
    if fixed is None or moving is None:
        raise ValueError("Either fixed and moving points must be provided, or tag_file must be specified.")

    R, T, rmse = orthogonal_procrustes(fixed, moving)
    print(f"Rotation Matrix (R): \n{R}\n")
    print(f"Translation Vector (T): \n{T}\n")

    ppe = []
    per_point_error = compute_per_point_error(fixed, moving, R, T)
    print("\nFRE for each point:")
    for i, fre in enumerate(per_point_error):
        ppe.append(fre)
        print(f"Point {i}: {fre:.6f} mm")

    print(f"\nRMSE: {rmse:.6f} mm")

    write_transform_file(filename, R, T)

    return R, T, rmse, ppe

def orthogonal_procrustes(fixed, moving):
    """
    Implements point-based registration via the Orthogonal Procrustes method.

    :param fixed: Point set, N x 3 ndarray
    :param moving: Point set, N x 3 ndarray of corresponding points
    :returns: 3x3 rotation ndarray, 3x1 translation ndarray, FRE
    :raises: ValueError
    """
    # Compute means
    p = np.mean(moving, axis=0)
    p_prime = np.mean(fixed, axis=0)

    # Center the points
    q = moving - p
    q_prime = fixed - p_prime

    # Compute the covariance matrix
    H = np.dot(q.T, q_prime)

    # # Add regularization to the diagonal of H
    # H += 1e-10 * np.eye(H.shape[0])

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T

    # Compute rotation matrix
    R = np.dot(V, U.T)
    
    # Correct for reflections
    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = np.dot(V, U.T)

    # Compute translation vector
    T = p_prime - np.dot(R, p)

    # Apply the transformation to the moving points
    transformed_moving = np.dot(moving, R.T) + T

    # Compute the error as the distance between transformed moving points and fixed points
    error = np.linalg.norm(fixed - transformed_moving, axis=1)
    rmse = np.sqrt(np.mean(error**2))

    return R, T, rmse

def compute_per_point_error(fixed, moving, R, T):
    """
    Computes the registration error for each point.
    
    :param fixed: Fixed point set, N x 3 ndarray
    :param moving: Moving point set, N x 3 ndarray
    :param R: Rotation matrix, 3x3 ndarray
    :param T: Translation vector, 3x1 ndarray
    :return: Array of per-point errors
    """
    # Apply the transformation to the moving points
    transformed_moving = np.dot(moving, R.T) + T

    # Compute the error as the distance between transformed moving points and fixed points
    error = np.linalg.norm(fixed - transformed_moving, axis=1)
    
    return error

def write_transform_file(filename, R, T):
    """
    Writes the transformation matrix to a .xfm file in the specified format.
    
    :param filename: Name of the file to save the transformation
    :param R: Rotation matrix, 3x3 ndarray
    :param T: Translation vector, 3x1 ndarray
    """
    with open(filename, 'w') as f:
        f.write("MNI Transform File\n")
        f.write("% Single linear transformation.\n\n")
        f.write("Transform_Type = Linear;\n")
        f.write("Linear_Transform =\n")
        for i in range(3):
            f.write(f"{R[i, 0]:.6f} {R[i, 1]:.6f} {R[i, 2]:.6f} {T[i]:.6f}\n")
        f.write(";\n")


if __name__ == "__main__":

    # R, T, _ = register(fixed_points, moving_points)
    R, T, _ = register(tag_file='no_kalman_5cam_offset_2.2rms.tag')

    # transformed_points = transform(moving_points, R, T)
    transformed_points = apply_transform(moving_points, xfm_file='transformation.xfm')
    print(f"\nTransformed Points: \n {transformed_points}")
    print(f"\nDifference between Transformed Moving Points and Fixed Points:\n{fixed_points - transformed_points}")