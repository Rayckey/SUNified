import h5py
import numpy as np
import json
import os
from PIL import Image
import glob

from kuafu.paths import external_artifact_root

def load_dataset(directory_path):
    """
    Load and analyze dataset from IsaacLab outputs directory.
    Prints type and shape information for all data files.
    """
    print(f"Loading dataset from: {directory_path}")
    print("=" * 60)
    
    # Load HDF5 state file
    state_file = os.path.join(directory_path, "state.h5")
    if os.path.exists(state_file):
        print("\n📊 HDF5 State File:")
        print("-" * 30)
        with h5py.File(state_file, 'r') as f:
            print(f"Keys: {list(f.keys())}")
            for key in f.keys():
                data = f[key]
                if hasattr(data, 'shape'):
                    print(f"  {key}: type={type(data).__name__}, shape={data.shape}, dtype={data.dtype}")
                else:
                    print(f"  {key}: type={type(data).__name__}")
    
    # Load RGB images
    rgb_files = sorted(glob.glob(os.path.join(directory_path, "rgb_*.png")))
    if rgb_files:
        print(f"\n🖼️  RGB Images ({len(rgb_files)} files):")
        print("-" * 30)
        # Sample first image to get dimensions
        sample_img = Image.open(rgb_files[0])
        img_array = np.array(sample_img)
        print(f"  Sample: {os.path.basename(rgb_files[0])}")
        print(f"  Type: {type(img_array).__name__}")
        print(f"  Shape: {img_array.shape}")
        print(f"  Dtype: {img_array.dtype}")
        print(f"  Total RGB files: {len(rgb_files)}")
    
    # Load instance segmentation images
    seg_files = sorted(glob.glob(os.path.join(directory_path, "instance_segmentation_*.png")))
    if seg_files:
        print(f"\n🎯 Instance Segmentation Images ({len(seg_files)} files):")
        print("-" * 30)
        # Sample first segmentation image
        sample_seg = Image.open(seg_files[0])
        seg_array = np.array(sample_seg)
        print(f"  Sample: {os.path.basename(seg_files[0])}")
        print(f"  Type: {type(seg_array).__name__}")
        print(f"  Shape: {seg_array.shape}")
        print(f"  Dtype: {seg_array.dtype}")
        print(f"  Total segmentation files: {len(seg_files)}")
    
    # Load distance to image plane data
    distance_files = sorted(glob.glob(os.path.join(directory_path, "distance_to_image_plane_*.npy")))
    if distance_files:
        print(f"\n📏 Distance to Image Plane Data ({len(distance_files)} files):")
        print("-" * 30)
        # Sample first distance file
        sample_distance = np.load(distance_files[0])
        print(f"  Sample: {os.path.basename(distance_files[0])}")
        print(f"  Type: {type(sample_distance).__name__}")
        print(f"  Shape: {sample_distance.shape}")
        print(f"  Dtype: {sample_distance.dtype}")
        print(f"  Total distance files: {len(distance_files)}")
    
    # Load instance segmentation mapping files
    mapping_files = sorted(glob.glob(os.path.join(directory_path, "instance_segmentation_mapping_*.json")))
    if mapping_files:
        print(f"\n🗺️  Instance Segmentation Mappings ({len(mapping_files)} files):")
        print("-" * 30)
        # Sample first mapping file
        with open(mapping_files[0], 'r') as f:
            sample_mapping = json.load(f)
        print(f"  Sample: {os.path.basename(mapping_files[0])}")
        print(f"  Type: {type(sample_mapping).__name__}")
        print(f"  Keys: {len(sample_mapping)} mappings")
        print(f"  Sample mapping: {list(sample_mapping.items())[0] if sample_mapping else 'None'}")
        print(f"  Total mapping files: {len(mapping_files)}")
    
    # Load semantics mapping files
    semantics_files = sorted(glob.glob(os.path.join(directory_path, "instance_segmentation_semantics_mapping_*.json")))
    if semantics_files:
        print(f"\n🏷️  Semantics Mappings ({len(semantics_files)} files):")
        print("-" * 30)
        # Sample first semantics file
        with open(semantics_files[0], 'r') as f:
            sample_semantics = json.load(f)
        print(f"  Sample: {os.path.basename(semantics_files[0])}")
        print(f"  Type: {type(sample_semantics).__name__}")
        print(f"  Keys: {len(sample_semantics)} mappings")
        print(f"  Sample mapping: {list(sample_semantics.items())[0] if sample_semantics else 'None'}")
        print(f"  Total semantics files: {len(semantics_files)}")
    
    print("\n" + "=" * 60)
    print("Dataset loading complete!")

def load_specific_frame(directory_path, frame_number):
    """
    Load all data for a specific frame number.
    Returns a dictionary with all available data for that frame.
    """
    frame_data = {}
    
    # Load RGB image
    rgb_file = os.path.join(directory_path, f"rgb_{frame_number:03d}_000.png")
    if os.path.exists(rgb_file):
        frame_data['rgb'] = np.array(Image.open(rgb_file))
    
    # Load instance segmentation
    seg_file = os.path.join(directory_path, f"instance_segmentation_{frame_number:03d}_000.png")
    if os.path.exists(seg_file):
        frame_data['instance_segmentation'] = np.array(Image.open(seg_file))
    
    # Load distance data
    distance_file = os.path.join(directory_path, f"distance_to_image_plane_{frame_number:03d}_000.npy")
    if os.path.exists(distance_file):
        frame_data['distance'] = np.load(distance_file)
    
    # Load mappings
    mapping_file = os.path.join(directory_path, f"instance_segmentation_mapping_{frame_number:03d}_000.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            frame_data['mapping'] = json.load(f)
    
    # Load semantics
    semantics_file = os.path.join(directory_path, f"instance_segmentation_semantics_mapping_{frame_number:03d}_000.json")
    if os.path.exists(semantics_file):
        with open(semantics_file, 'r') as f:
            frame_data['semantics'] = json.load(f)
    
    return frame_data

if __name__ == "__main__":
    # Load the entire dataset overview
    dataset_path = str(external_artifact_root() / "debug" / "sensor_dataset")
    load_dataset(dataset_path)
    
    print("\n" + "=" * 60)
    print("Loading specific frame example (frame 480):")
    print("=" * 60)
    
    # Example: Load specific frame
    frame_data = load_specific_frame(dataset_path, 480)
    for key, value in frame_data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: type={type(value).__name__}, shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: type={type(value).__name__}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")