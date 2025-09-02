import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from collections import defaultdict

def find_matching_pairs(data_dir='./data', validation_dir='./validation', shuffle=True, max_train=None, max_val=None):
    """Find matching pressure-velocity file pairs in data and validation directories
    
    Args:
        data_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        shuffle (bool): Whether to shuffle the file pairs
        max_train (int): Maximum number of training samples to use
        max_val (int): Maximum number of validation samples to use
        
    Returns:
        tuple: Lists of training and validation file pairs
    """
    # Regular expression to extract the numeric part
    pattern = r'(\d{7})'
    
    # Find all pressure files in the directories
    pressure_files = glob.glob(os.path.join(data_dir, 'pre_slice_fld_*.bin'))
    val_pressure_files = glob.glob(os.path.join(validation_dir, 'pre_slice_fld_*.bin'))
    
    # Match files by their numeric part
    train_pairs = []
    for p_file in pressure_files:
        p_match = re.search(pattern, p_file)
        if p_match:
            p_num = p_match.group(1)
            v_file = os.path.join(data_dir, f"vex_slice_fld_{p_num}.bin")
            if os.path.exists(v_file):
                train_pairs.append((p_file, v_file))
    
    val_pairs = []
    for p_file in val_pressure_files:
        p_match = re.search(pattern, p_file)
        if p_match:
            p_num = p_match.group(1)
            v_file = os.path.join(validation_dir, f"vex_slice_fld_{p_num}.bin")
            if os.path.exists(v_file):
                val_pairs.append((p_file, v_file))
    
    # Shuffle the pairs if requested
    if shuffle:
        random.shuffle(train_pairs)
        random.shuffle(val_pairs)
    
    # Limit the number of pairs if requested
    if max_train is not None and len(train_pairs) > max_train:
        train_pairs = train_pairs[:max_train]
    
    if max_val is not None and len(val_pairs) > max_val:
        val_pairs = val_pairs[:max_val]
    
    print(f"Using {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")
    
    return train_pairs, val_pairs

def find_matching_triplets(data_dir='./data', validation_dir='./validation', shuffle=True, max_train=None, max_val=None):
    """Find matching pressure-velocity-wall_shear_stress file triplets in data and validation directories
    
    Args:
        data_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        shuffle (bool): Whether to shuffle the file triplets
        max_train (int): Maximum number of training samples to use
        max_val (int): Maximum number of validation samples to use
        
    Returns:
        tuple: Lists of training and validation file triplets
    """
    # Regular expression to extract the numeric part
    pattern = r'(\d{7})'
    
    # Find all pressure files in the directories
    pressure_files = glob.glob(os.path.join(data_dir, 'pre_slice_fld_*.bin'))
    val_pressure_files = glob.glob(os.path.join(validation_dir, 'pre_slice_fld_*.bin'))
    
    # Match files by their numeric part for training
    train_triplets = []
    for p_file in pressure_files:
        p_match = re.search(pattern, p_file)
        if p_match:
            p_num = p_match.group(1)
            v_file = os.path.join(data_dir, f"vex_slice_fld_{p_num}.bin")
            # Assuming wall shear stress files follow a similar naming pattern
            wss_file = os.path.join(data_dir, f"wss_slice_fld_{p_num}.bin")
            
            if os.path.exists(v_file) and os.path.exists(wss_file):
                train_triplets.append((p_file, wss_file, v_file))
    
    # Match files for validation
    val_triplets = []
    for p_file in val_pressure_files:
        p_match = re.search(pattern, p_file)
        if p_match:
            p_num = p_match.group(1)
            v_file = os.path.join(validation_dir, f"vex_slice_fld_{p_num}.bin")
            wss_file = os.path.join(validation_dir, f"wss_slice_fld_{p_num}.bin")
            
            if os.path.exists(v_file) and os.path.exists(wss_file):
                val_triplets.append((p_file, wss_file, v_file))
    
    # Shuffle the triplets if requested
    if shuffle:
        random.shuffle(train_triplets)
        random.shuffle(val_triplets)
    
    # Limit the number of triplets if requested
    if max_train is not None and len(train_triplets) > max_train:
        train_triplets = train_triplets[:max_train]
    
    if max_val is not None and len(val_triplets) > max_val:
        val_triplets = val_triplets[:max_val]
    
    print(f"Using {len(train_triplets)} training triplets and {len(val_triplets)} validation triplets")
    
    return train_triplets, val_triplets

def find_matching_quadruplets(data_dir='./data', validation_dir='./validation', shuffle=True, max_train=None, max_val=None):
    """Find matching pressure-wall_shear_stress-u_velocity-v_velocity file quadruplets
    
    Args:
        data_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        shuffle (bool): Whether to shuffle the file quadruplets
        max_train (int): Maximum number of training samples to use
        max_val (int): Maximum number of validation samples to use
        
    Returns:
        tuple: Lists of training and validation file quadruplets (pressure, wss, u_vel, v_vel)
    """
    # Regular expression to extract the numeric part
    pattern = r'(\d{7})'
    
    # Find all pressure files in the directories
    pressure_files = glob.glob(os.path.join(data_dir, 'pre_slice_fld_*.bin'))
    val_pressure_files = glob.glob(os.path.join(validation_dir, 'pre_slice_fld_*.bin'))
    
    # Match files by their numeric part for training
    train_quadruplets = []
    for p_file in pressure_files:
        p_match = re.search(pattern, p_file)
        if p_match:
            p_num = p_match.group(1)
            wss_file = os.path.join(data_dir, f"wss_slice_fld_{p_num}.bin")
            u_file = os.path.join(data_dir, f"vex_slice_fld_{p_num}.bin")  # streamwise velocity
            v_file = os.path.join(data_dir, f"vez_slice_fld_{p_num}.bin")  # wall-normal velocity
            
            if os.path.exists(wss_file) and os.path.exists(u_file) and os.path.exists(v_file):
                train_quadruplets.append((p_file, wss_file, u_file, v_file))
    
    # Match files for validation
    val_quadruplets = []
    for p_file in val_pressure_files:
        p_match = re.search(pattern, p_file)
        if p_match:
            p_num = p_match.group(1)
            wss_file = os.path.join(validation_dir, f"wss_slice_fld_{p_num}.bin")
            u_file = os.path.join(validation_dir, f"vex_slice_fld_{p_num}.bin")  # streamwise velocity
            v_file = os.path.join(validation_dir, f"vez_slice_fld_{p_num}.bin")  # wall-normal velocity
            
            if os.path.exists(wss_file) and os.path.exists(u_file) and os.path.exists(v_file):
                val_quadruplets.append((p_file, wss_file, u_file, v_file))
    
    # Shuffle the quadruplets if requested
    if shuffle:
        random.shuffle(train_quadruplets)
        random.shuffle(val_quadruplets)
    
    # Limit the number of quadruplets if requested
    if max_train is not None and len(train_quadruplets) > max_train:
        train_quadruplets = train_quadruplets[:max_train]
    
    if max_val is not None and len(val_quadruplets) > max_val:
        val_quadruplets = val_quadruplets[:max_val]
    
    print(f"Using {len(train_quadruplets)} training quadruplets and {len(val_quadruplets)} validation quadruplets")
    
    return train_quadruplets, val_quadruplets

def read_binary_slice(filename, shape=(64, 64), dtype='float64', transpose=True):
    """Read a binary slice file and reshape appropriately with error handling"""
    try:
        # Check if file exists
        if not os.path.exists(filename):
            return None, f"File does not exist"
            
        # Check file size
        file_size = os.path.getsize(filename)
        expected_size = np.prod(shape) * np.dtype(dtype).itemsize
        
        if file_size != expected_size:
            return None, f"File size mismatch: expected {expected_size} bytes, got {file_size} bytes"
        
        # Read the data
        data = np.fromfile(filename, dtype=dtype)
        
        if len(data) == 0:
            return None, "Empty file (no data read)"
            
        try:
            data = data.reshape(shape, order='F')  # Fortran-style ordering
        except ValueError as e:
            return None, f"Reshape error: {e}"
        
        if transpose:
            data = data.T
        
        # Check for NaN or Inf values
        if np.isnan(data).any():
            return None, "Contains NaN values"
        
        if np.isinf(data).any():
            return None, "Contains Inf values"
            
        return data, None  # No error
        
    except Exception as e:
        return None, f"Exception: {e}"

def compute_dataset_statistics(file_pairs, sample_size=1000, shape=(64, 64), dtype='float64'):
    """
    Compute dataset statistics using a sample of files for proper normalization
    
    Args:
        file_pairs (list): List of (pressure_file, velocity_file) tuples
        sample_size (int): Number of files to sample for statistics
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        
    Returns:
        dict: Dictionary with normalization parameters
    """
    print(f"Computing dataset statistics from {min(sample_size, len(file_pairs))} samples...")
    
    # Sample file pairs if we have more than sample_size
    if len(file_pairs) > sample_size:
        sampled_pairs = random.sample(file_pairs, sample_size)
    else:
        sampled_pairs = file_pairs
    
    # Initialize arrays to store statistics
    p_mins, p_maxs = [], []
    v_mins, v_maxs = [], []
    
    # Process each pair
    for p_file, v_file in tqdm.tqdm(sampled_pairs):
        # Read pressure file
        p_data, p_error = read_binary_slice(p_file, shape, dtype)
        if p_error is None:
            p_mins.append(np.min(p_data))
            p_maxs.append(np.max(p_data))
        
        # Read velocity file
        v_data, v_error = read_binary_slice(v_file, shape, dtype)
        if v_error is None:
            v_mins.append(np.min(v_data))
            v_maxs.append(np.max(v_data))
    
    # Compute global statistics
    stats = {
        'pressure_min': np.min(p_mins),
        'pressure_max': np.max(p_maxs),
        'velocity_min': np.min(v_mins),
        'velocity_max': np.max(v_maxs)
    }
    
    # Print the statistics
    print("Dataset statistics:")
    print(f"  Pressure range: {stats['pressure_min']:.6f} to {stats['pressure_max']:.6f}")
    print(f"  Velocity range: {stats['velocity_min']:.6f} to {stats['velocity_max']:.6f}")
    
    return stats

def compute_dataset_statistics_triplets(file_triplets, sample_size=1000, shape=(64, 64), dtype='float64'):
    """
    Compute dataset statistics for pressure, wall shear stress, and velocity fields
    
    Args:
        file_triplets (list): List of (pressure_file, wss_file, velocity_file) tuples
        sample_size (int): Number of files to sample for statistics
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        
    Returns:
        dict: Dictionary with normalization parameters
    """
    print(f"Computing dataset statistics from {min(sample_size, len(file_triplets))} samples...")
    
    # Sample file triplets if we have more than sample_size
    if len(file_triplets) > sample_size:
        sampled_triplets = random.sample(file_triplets, sample_size)
    else:
        sampled_triplets = file_triplets
    
    # Initialize arrays to store statistics
    p_mins, p_maxs = [], []
    wss_mins, wss_maxs = [], []
    v_mins, v_maxs = [], []
    
    # Process each triplet
    for p_file, wss_file, v_file in tqdm.tqdm(sampled_triplets):
        # Read pressure file
        p_data, p_error = read_binary_slice(p_file, shape, dtype)
        if p_error is None:
            p_mins.append(np.min(p_data))
            p_maxs.append(np.max(p_data))
        
        # Read wall shear stress file
        wss_data, wss_error = read_binary_slice(wss_file, shape, dtype)
        if wss_error is None:
            wss_mins.append(np.min(wss_data))
            wss_maxs.append(np.max(wss_data))
        
        # Read velocity file
        v_data, v_error = read_binary_slice(v_file, shape, dtype)
        if v_error is None:
            v_mins.append(np.min(v_data))
            v_maxs.append(np.max(v_data))
    
    # Compute global statistics
    stats = {
        'pressure_min': np.min(p_mins),
        'pressure_max': np.max(p_maxs),
        'wss_min': np.min(wss_mins),
        'wss_max': np.max(wss_maxs),
        'velocity_min': np.min(v_mins),
        'velocity_max': np.max(v_maxs)
    }
    
    # Print the statistics
    print("Dataset statistics:")
    print(f"  Pressure range: {stats['pressure_min']:.6f} to {stats['pressure_max']:.6f}")
    print(f"  Wall Shear Stress range: {stats['wss_min']:.6f} to {stats['wss_max']:.6f}")
    print(f"  Velocity range: {stats['velocity_min']:.6f} to {stats['velocity_max']:.6f}")
    
    return stats

def compute_dataset_statistics_quadruplets(file_quadruplets, sample_size=1000, shape=(64, 64), dtype='float64'):
    """
    Compute dataset statistics for pressure, wall shear stress, u-velocity, and v-velocity fields
    
    Args:
        file_quadruplets (list): List of (pressure_file, wss_file, u_velocity_file, v_velocity_file) tuples
        sample_size (int): Number of files to sample for statistics
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        
    Returns:
        dict: Dictionary with normalization parameters
    """
    print(f"Computing dataset statistics from {min(sample_size, len(file_quadruplets))} samples...")
    
    # Sample file quadruplets if we have more than sample_size
    if len(file_quadruplets) > sample_size:
        sampled_quadruplets = random.sample(file_quadruplets, sample_size)
    else:
        sampled_quadruplets = file_quadruplets
    
    # Initialize arrays to store statistics
    p_mins, p_maxs = [], []
    wss_mins, wss_maxs = [], []
    u_mins, u_maxs = [], []
    v_mins, v_maxs = [], []
    
    # Process each quadruplet
    for p_file, wss_file, u_file, v_file in tqdm.tqdm(sampled_quadruplets):
        # Read pressure file
        p_data, p_error = read_binary_slice(p_file, shape, dtype)
        if p_error is None:
            p_mins.append(np.min(p_data))
            p_maxs.append(np.max(p_data))
        
        # Read wall shear stress file
        wss_data, wss_error = read_binary_slice(wss_file, shape, dtype)
        if wss_error is None:
            wss_mins.append(np.min(wss_data))
            wss_maxs.append(np.max(wss_data))
        
        # Read u-velocity file
        u_data, u_error = read_binary_slice(u_file, shape, dtype)
        if u_error is None:
            u_mins.append(np.min(u_data))
            u_maxs.append(np.max(u_data))
        
        # Read v-velocity file
        v_data, v_error = read_binary_slice(v_file, shape, dtype)
        if v_error is None:
            v_mins.append(np.min(v_data))
            v_maxs.append(np.max(v_data))
    
    # Compute global statistics
    stats = {
        'pressure_min': np.min(p_mins),
        'pressure_max': np.max(p_maxs),
        'wss_min': np.min(wss_mins),
        'wss_max': np.max(wss_maxs),
        'u_velocity_min': np.min(u_mins),
        'u_velocity_max': np.max(u_maxs),
        'v_velocity_min': np.min(v_mins),
        'v_velocity_max': np.max(v_maxs)
    }
    
    # Print the statistics
    print("Dataset statistics:")
    print(f"  Pressure range: {stats['pressure_min']:.6f} to {stats['pressure_max']:.6f}")
    print(f"  Wall Shear Stress range: {stats['wss_min']:.6f} to {stats['wss_max']:.6f}")
    print(f"  U-Velocity range: {stats['u_velocity_min']:.6f} to {stats['u_velocity_max']:.6f}")
    print(f"  V-Velocity range: {stats['v_velocity_min']:.6f} to {stats['v_velocity_max']:.6f}")
    
    return stats

def normalize_data(data, data_min, data_max, target_min=-1, target_max=1):
    """
    Normalize data to a target range with robust handling
    
    Args:
        data (numpy.ndarray): Data to normalize
        data_min (float): Minimum value in the data range
        data_max (float): Maximum value in the data range
        target_min (float): Minimum value in the target range
        target_max (float): Maximum value in the target range
        
    Returns:
        numpy.ndarray: Normalized data
    """
    # Check for valid range to avoid division by zero
    if np.isclose(data_max, data_min):
        # If the range is effectively zero, return centered data
        return np.full_like(data, (target_max + target_min) / 2)
    
    # Clip the input data to the expected range
    data_clipped = np.clip(data, data_min, data_max)
    
    # Normalize to [0, 1]
    normalized = (data_clipped - data_min) / (data_max - data_min)
    
    # Scale to target range
    return normalized * (target_max - target_min) + target_min

class PressureVelocityDataset(Dataset):
    """PyTorch dataset for pressure-velocity pairs"""
    
    def __init__(self, file_pairs, shape=(64, 64), dtype='float64', normalize=True):
        """
        Initialize the dataset
        
        Args:
            file_pairs (list): List of (pressure_file, velocity_file) tuples
            shape (tuple): Expected shape of each slice
            dtype (str): Data type
            normalize (bool): Whether to normalize the data
        """
        self.file_pairs = file_pairs
        self.shape = shape
        self.dtype = dtype
        self.normalize = normalize
        
        # Compute dataset statistics for normalization if needed
        if normalize:
            self.stats = compute_dataset_statistics(
                file_pairs, 
                sample_size=min(1000, len(file_pairs)),
                shape=shape,
                dtype=dtype
            )
        else:
            self.stats = None
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        p_file, v_file = self.file_pairs[idx]
        
        # Read the files
        p_data, p_error = read_binary_slice(p_file, self.shape, self.dtype)
        v_data, v_error = read_binary_slice(v_file, self.shape, self.dtype)
        
        # Handle errors
        if p_error is not None or v_error is not None:
            # Return zeros for invalid files
            p_data = np.zeros(self.shape, dtype=np.float32)
            v_data = np.zeros(self.shape, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize and self.stats is not None:
            p_data = normalize_data(
                p_data, 
                self.stats['pressure_min'], 
                self.stats['pressure_max']
            )
            v_data = normalize_data(
                v_data, 
                self.stats['velocity_min'], 
                self.stats['velocity_max']
            )
        
        # Add channel dimension and convert to float32
        p_data = p_data.reshape(1, *self.shape).astype(np.float32)
        v_data = v_data.reshape(1, *self.shape).astype(np.float32)
        
        # Convert to PyTorch tensors
        p_tensor = torch.tensor(p_data, dtype=torch.float32)
        v_tensor = torch.tensor(v_data, dtype=torch.float32)
        
        return p_tensor, v_tensor

class PressureWSSVelocityDataset(Dataset):
    """PyTorch dataset for pressure-wall_shear_stress-velocity triplets"""
    
    def __init__(self, file_triplets, shape=(64, 64), dtype='float64', normalize=True):
        """
        Initialize the dataset
        
        Args:
            file_triplets (list): List of (pressure_file, wss_file, velocity_file) tuples
            shape (tuple): Expected shape of each slice
            dtype (str): Data type
            normalize (bool): Whether to normalize the data
        """
        self.file_triplets = file_triplets
        self.shape = shape
        self.dtype = dtype
        self.normalize = normalize
        
        # Compute dataset statistics for normalization if needed
        if normalize:
            self.stats = compute_dataset_statistics_triplets(
                file_triplets, 
                sample_size=min(1000, len(file_triplets)),
                shape=shape,
                dtype=dtype
            )
        else:
            self.stats = None
    
    def __len__(self):
        return len(self.file_triplets)
    
    def __getitem__(self, idx):
        p_file, wss_file, v_file = self.file_triplets[idx]
        
        # Read the files
        p_data, p_error = read_binary_slice(p_file, self.shape, self.dtype)
        wss_data, wss_error = read_binary_slice(wss_file, self.shape, self.dtype)
        v_data, v_error = read_binary_slice(v_file, self.shape, self.dtype)
        
        # Handle errors
        if p_error is not None or wss_error is not None or v_error is not None:
            # Return zeros for invalid files
            p_data = np.zeros(self.shape, dtype=np.float32)
            wss_data = np.zeros(self.shape, dtype=np.float32)
            v_data = np.zeros(self.shape, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize and self.stats is not None:
            p_data = normalize_data(
                p_data, 
                self.stats['pressure_min'], 
                self.stats['pressure_max']
            )
            wss_data = normalize_data(
                wss_data, 
                self.stats['wss_min'], 
                self.stats['wss_max']
            )
            v_data = normalize_data(
                v_data, 
                self.stats['velocity_min'], 
                self.stats['velocity_max']
            )
        
        # Combine pressure and wall shear stress into a 2-channel input
        # Shape: (2, height, width)
        input_data = np.stack([p_data, wss_data], axis=0).astype(np.float32)
        
        # Add channel dimension to velocity and convert to float32
        v_data = v_data.reshape(1, *self.shape).astype(np.float32)
        
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        v_tensor = torch.tensor(v_data, dtype=torch.float32)
        
        return input_tensor, v_tensor

class PressureWSSUVVelocityDataset(Dataset):
    """PyTorch dataset for pressure-wall_shear_stress-u_velocity-v_velocity quadruplets"""
    
    def __init__(self, file_quadruplets, shape=(64, 64), dtype='float64', normalize=True):
        """
        Initialize the dataset
        
        Args:
            file_quadruplets (list): List of (pressure_file, wss_file, u_velocity_file, v_velocity_file) tuples
            shape (tuple): Expected shape of each slice
            dtype (str): Data type
            normalize (bool): Whether to normalize the data
        """
        self.file_quadruplets = file_quadruplets
        self.shape = shape
        self.dtype = dtype
        self.normalize = normalize
        
        # Compute dataset statistics for normalization if needed
        if normalize:
            self.stats = compute_dataset_statistics_quadruplets(
                file_quadruplets, 
                sample_size=min(1000, len(file_quadruplets)),
                shape=shape,
                dtype=dtype
            )
        else:
            self.stats = None
    
    def __len__(self):
        return len(self.file_quadruplets)
    
    def __getitem__(self, idx):
        p_file, wss_file, u_file, v_file = self.file_quadruplets[idx]
        
        # Read the files
        p_data, p_error = read_binary_slice(p_file, self.shape, self.dtype)
        wss_data, wss_error = read_binary_slice(wss_file, self.shape, self.dtype)
        u_data, u_error = read_binary_slice(u_file, self.shape, self.dtype)
        v_data, v_error = read_binary_slice(v_file, self.shape, self.dtype)
        
        # Handle errors
        if p_error is not None or wss_error is not None or u_error is not None or v_error is not None:
            # Return zeros for invalid files
            p_data = np.zeros(self.shape, dtype=np.float32)
            wss_data = np.zeros(self.shape, dtype=np.float32)
            u_data = np.zeros(self.shape, dtype=np.float32)
            v_data = np.zeros(self.shape, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize and self.stats is not None:
            p_data = normalize_data(
                p_data, 
                self.stats['pressure_min'], 
                self.stats['pressure_max']
            )
            wss_data = normalize_data(
                wss_data, 
                self.stats['wss_min'], 
                self.stats['wss_max']
            )
            u_data = normalize_data(
                u_data, 
                self.stats['u_velocity_min'], 
                self.stats['u_velocity_max']
            )
            v_data = normalize_data(
                v_data, 
                self.stats['v_velocity_min'], 
                self.stats['v_velocity_max']
            )
        
        # Combine pressure and wall shear stress into a 2-channel input
        # Shape: (2, height, width)
        input_data = np.stack([p_data, wss_data], axis=0).astype(np.float32)
        
        # Combine u and v velocities into a 2-channel output
        # Shape: (2, height, width)
        velocity_data = np.stack([u_data, v_data], axis=0).astype(np.float32)
        
        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        velocity_tensor = torch.tensor(velocity_data, dtype=torch.float32)
        
        return input_tensor, velocity_tensor

def create_dataloaders(file_pairs, batch_size=64, shape=(64, 64), 
                     dtype='float64', normalize=True, num_workers=4, 
                     shuffle=True, pin_memory=True, val_pairs=None, val_batch_size=None):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        file_pairs (list): List of (pressure_file, velocity_file) tuples for training
        batch_size (int): Batch size for training
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        normalize (bool): Whether to normalize the data
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer
        val_pairs (list): List of (pressure_file, velocity_file) tuples for validation
        val_batch_size (int): Batch size for validation (defaults to same as training)
        
    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoaders for training and validation
    """
    # Create training dataset
    train_dataset = PressureVelocityDataset(
        file_pairs,
        shape=shape,
        dtype=dtype,
        normalize=normalize
    )
    
    # Create training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    # Create validation data loader if validation pairs are provided
    val_loader = None
    if val_pairs is not None:
        val_dataset = PressureVelocityDataset(
            val_pairs,
            shape=shape,
            dtype=dtype,
            normalize=normalize
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
    
    return train_loader, val_loader

def create_dataloaders_triplets(file_triplets, batch_size=64, shape=(64, 64), 
                               dtype='float64', normalize=True, num_workers=4, 
                               shuffle=True, pin_memory=True, val_triplets=None, val_batch_size=None):
    """
    Create PyTorch DataLoaders for training and validation with triplet data
    
    Args:
        file_triplets (list): List of (pressure_file, wss_file, velocity_file) tuples for training
        batch_size (int): Batch size for training
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        normalize (bool): Whether to normalize the data
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer
        val_triplets (list): List of (pressure_file, wss_file, velocity_file) tuples for validation
        val_batch_size (int): Batch size for validation (defaults to same as training)
        
    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoaders for training and validation
    """
    # Create training dataset
    train_dataset = PressureWSSVelocityDataset(
        file_triplets,
        shape=shape,
        dtype=dtype,
        normalize=normalize
    )
    
    # Create training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    # Create validation data loader if validation triplets are provided
    val_loader = None
    if val_triplets is not None:
        val_dataset = PressureWSSVelocityDataset(
            val_triplets,
            shape=shape,
            dtype=dtype,
            normalize=normalize
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
    
    return train_loader, val_loader

def create_dataloaders_quadruplets(file_quadruplets, batch_size=64, shape=(64, 64), 
                                  dtype='float64', normalize=True, num_workers=4, 
                                  shuffle=True, pin_memory=True, val_quadruplets=None, val_batch_size=None):
    """
    Create PyTorch DataLoaders for training and validation with quadruplet data (pressure, WSS, u-vel, v-vel)
    
    Args:
        file_quadruplets (list): List of (pressure_file, wss_file, u_velocity_file, v_velocity_file) tuples for training
        batch_size (int): Batch size for training
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        normalize (bool): Whether to normalize the data
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer
        val_quadruplets (list): List of (pressure_file, wss_file, u_velocity_file, v_velocity_file) tuples for validation
        val_batch_size (int): Batch size for validation (defaults to same as training)
        
    Returns:
        tuple: (train_loader, val_loader) - PyTorch DataLoaders for training and validation
    """
    # Create training dataset
    train_dataset = PressureWSSUVVelocityDataset(
        file_quadruplets,
        shape=shape,
        dtype=dtype,
        normalize=normalize
    )
    
    # Create training data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True
    )
    
    # Create validation data loader if validation quadruplets are provided
    val_loader = None
    if val_quadruplets is not None:
        val_dataset = PressureWSSUVVelocityDataset(
            val_quadruplets,
            shape=shape,
            dtype=dtype,
            normalize=normalize
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size or batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available()
        )
    
    return train_loader, val_loader

def load_validation_data_numpy(val_pairs, max_samples=None, shape=(64, 64), 
                              dtype='float64', normalize=True):
    """
    Load validation data as NumPy arrays
    
    Args:
        val_pairs (list): List of validation file pairs
        max_samples (int): Maximum number of samples to load
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        normalize (bool): Whether to normalize the data
        
    Returns:
        tuple: (pressure_data, velocity_data) arrays for validation
    """
    if not val_pairs:
        return None, None
    
    # Limit the number of samples if requested
    if max_samples is not None:
        val_pairs = val_pairs[:min(max_samples, len(val_pairs))]
    
    # Create a dataset
    val_dataset = PressureVelocityDataset(
        val_pairs,
        shape=shape,
        dtype=dtype,
        normalize=normalize
    )
    
    # Create data loader with batch size equal to the whole dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=4
    )
    
    # Get the single batch containing all validation data
    for val_pressure, val_velocity in val_loader:
        return val_pressure.numpy(), val_velocity.numpy()
    
    return None, None

def load_validation_data_numpy_triplets(val_triplets, max_samples=None, shape=(64, 64), 
                                      dtype='float64', normalize=True):
    """
    Load validation data as NumPy arrays for triplets (pressure + WSS + velocity)
    
    Args:
        val_triplets (list): List of validation file triplets
        max_samples (int): Maximum number of samples to load
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        normalize (bool): Whether to normalize the data
        
    Returns:
        tuple: (input_data, velocity_data) arrays for validation
    """
    if not val_triplets:
        return None, None
    
    # Limit the number of samples if requested
    if max_samples is not None:
        val_triplets = val_triplets[:min(max_samples, len(val_triplets))]
    
    # Create a dataset
    val_dataset = PressureWSSVelocityDataset(
        val_triplets,
        shape=shape,
        dtype=dtype,
        normalize=normalize
    )
    
    # Create data loader with batch size equal to the whole dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=4
    )
    
    # Get the single batch containing all validation data
    for val_input, val_velocity in val_loader:
        return val_input.numpy(), val_velocity.numpy()
    
    return None, None

def load_validation_data_numpy_quadruplets(val_quadruplets, max_samples=None, shape=(64, 64), 
                                         dtype='float64', normalize=True):
    """
    Load validation data as NumPy arrays for quadruplets (pressure + WSS + u_velocity + v_velocity)
    
    Args:
        val_quadruplets (list): List of validation file quadruplets
        max_samples (int): Maximum number of samples to load
        shape (tuple): Expected shape of each slice
        dtype (str): Data type
        normalize (bool): Whether to normalize the data
        
    Returns:
        tuple: (input_data, velocity_data) arrays for validation
               input_data shape: (N, 2, H, W) - pressure and WSS
               velocity_data shape: (N, 2, H, W) - u and v velocities
    """
    if not val_quadruplets:
        return None, None
    
    # Limit the number of samples if requested
    if max_samples is not None:
        val_quadruplets = val_quadruplets[:min(max_samples, len(val_quadruplets))]
    
    # Create a dataset
    val_dataset = PressureWSSUVVelocityDataset(
        val_quadruplets,
        shape=shape,
        dtype=dtype,
        normalize=normalize
    )
    
    # Create data loader with batch size equal to the whole dataset
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=4
    )
    
    # Get the single batch containing all validation data
    for val_input, val_velocity in val_loader:
        return val_input.numpy(), val_velocity.numpy()
    
    return None, None