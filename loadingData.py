import os
import pickle as pkl  # Corrected import for Python 3
import numpy as np

def load_sequence(file_path):
    """Load the sequence data from the .pkl file with proper encoding."""
    with open(file_path, 'rb') as f:
        data = pkl.load(f, encoding='latin1')  # Use 'latin1' encoding to handle non-ASCII characters
    return data

def inspect_sequence(sequence_data):
    """Inspect the key contents of the sequence data."""
    print("Sequence Name:", sequence_data.get('sequence', 'N/A'))

    # Inspect the SMPL shape parameters (betas)
    if 'betas' in sequence_data:
        print("\nSMPL Shape Parameters (Betas):")
        for i, betas in enumerate(sequence_data['betas']):
            print(f"  Model {i}: {betas.shape} -> {betas}")

    # Inspect the SMPL body poses
    if 'poses' in sequence_data:
        print("\nSMPL Body Poses (30Hz):")
        for i, poses in enumerate(sequence_data['poses']):
            print(f"  Model {i}: {poses.shape} -> Frame 0: {poses[0]}")

    # Inspect the SMPL translations
    if 'trans' in sequence_data:
        print("\nSMPL Translations (30Hz):")
        for i, trans in enumerate(sequence_data['trans']):
            print(f"  Model {i}: {trans.shape} -> Frame 0: {trans[0]}")

    # Inspect the 2D joint detections
    if 'poses2D' in sequence_data:
        print("\n2D Joint Detections (Coco-Format):")
        for i, poses2D in enumerate(sequence_data['poses2D']):
            print(f"  Model {i}: {poses2D.shape} -> Frame 0: {poses2D[0]}")

    # Inspect the 3D joint positions
    if 'jointPositions' in sequence_data:
        print("\n3D Joint Positions:")
        for i, jointPositions in enumerate(sequence_data['jointPositions']):
            print(f"  Model {i}: {jointPositions.shape} -> Frame 0: {jointPositions[0]}")

    # Inspect camera poses
    if 'cam_poses' in sequence_data:
        print("\nCamera Poses (Extrinsics):")
        print(f"  {sequence_data['cam_poses'].shape} -> Frame 0: {sequence_data['cam_poses'][0]}")

    # Inspect camera intrinsics
    if 'cam_intrinsics' in sequence_data:
        print("\nCamera Intrinsics (K Matrix):")
        print(f"  {sequence_data['cam_intrinsics'].shape} -> {sequence_data['cam_intrinsics']}")

    # Inspect validity of camera poses
    if 'campose_valid' in sequence_data:
        # Check if it's a list
        if isinstance(sequence_data['campose_valid'], list):
            print(f"  List of length {len(sequence_data['campose_valid'])} -> Frame 0: {sequence_data['campose_valid'][0]}")
        else:
            print(f"  {sequence_data['campose_valid'].shape} -> Frame 0: {sequence_data['campose_valid'][0]}")

def main():
    # Path to the .pkl file (update this path to your file location)
    pkl_file_path = 'sequenceFiles/sequenceFiles/test/downtown_rampAndStairs_00.pkl'

    # Load the sequence data
    sequence_data = load_sequence(pkl_file_path)

    # Inspect the sequence data
    inspect_sequence(sequence_data)

if __name__ == "__main__":
    main()
