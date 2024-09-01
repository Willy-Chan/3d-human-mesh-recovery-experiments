import os
import pickle as pkl
import numpy as np

class ThreeDPWDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sequence_data = self.load_sequence(file_path)

    def load_sequence(self, file_path):
        """Load the sequence data from the .pkl file with proper encoding."""
        with open(file_path, 'rb') as f:
            data = pkl.load(f, encoding='latin1')
        return data

    def get_sequence_name(self):
        """Return the name of the sequence."""
        return self.sequence_data.get('sequence', 'N/A')

    def get_betas(self):
        """Return the SMPL shape parameters (betas)."""
        return self.sequence_data.get('betas', None)

    def get_poses(self):
        """Return the SMPL body poses (30Hz)."""
        return self.sequence_data.get('poses', None)

    def get_translations(self):
        """Return the SMPL translations (30Hz)."""
        return self.sequence_data.get('trans', None)

    def get_poses_2D(self):
        """Return the 2D joint detections (Coco-Format)."""
        return self.sequence_data.get('poses2D', None)

    def get_joint_positions_3D(self):
        """Return the 3D joint positions."""
        return self.sequence_data.get('jointPositions', None)

    def get_camera_poses(self):
        """Return the camera poses (extrinsics)."""
        return self.sequence_data.get('cam_poses', None)

    def get_camera_intrinsics(self):
        """Return the camera intrinsics (K matrix)."""
        return self.sequence_data.get('cam_intrinsics', None)

    def get_camera_pose_validity(self):
        """Return the camera pose validity."""
        return self.sequence_data.get('campose_valid', None)

    def print_info(self):
        """
        Print out all the functionalities of this class and their outputs.
        """
        print(f"Sequence Name: {self.get_sequence_name()}")
        
        betas = self.get_betas()
        if betas:
            print("\nSMPL Shape Parameters (Betas):")
            for i, beta in enumerate(betas):
                print(f"  Model {i}: {beta.shape} -> {beta}")

        poses = self.get_poses()
        if poses:
            print("\nSMPL Body Poses (30Hz):")
            for i, pose in enumerate(poses):
                print(f"  Model {i}: {pose.shape} -> Frame 0: {pose[0]}")

        translations = self.get_translations()
        if translations:
            print("\nSMPL Translations (30Hz):")
            for i, trans in enumerate(translations):
                print(f"  Model {i}: {trans.shape} -> Frame 0: {trans[0]}")

        poses_2D = self.get_poses_2D()
        if poses_2D:
            print("\n2D Joint Detections (Coco-Format):")
            for i, pose_2D in enumerate(poses_2D):
                print(f"  Model {i}: {pose_2D.shape} -> Frame 0: {pose_2D[0]}")

        joint_positions_3D = self.get_joint_positions_3D()
        if joint_positions_3D:
            print("\n3D Joint Positions:")
            for i, joint_position_3D in enumerate(joint_positions_3D):
                print(f"  Model {i}: {joint_position_3D.shape} -> Frame 0: {joint_position_3D[0]}")

        camera_poses = self.get_camera_poses()
        print("\nCamera Poses (Extrinsics):")
        print(f"  {camera_poses.shape} -> Frame 0: {camera_poses[0]}")
            
            

        camera_intrinsics = self.get_camera_intrinsics()
        print("\nCamera Intrinsics (K Matrix):")
        print(f"  {camera_intrinsics.shape} -> {camera_intrinsics}")

        camera_pose_validity = self.get_camera_pose_validity()
        if camera_pose_validity:
            if isinstance(camera_pose_validity, list):
                print(f"\nCamera Pose Validity: List of length {len(camera_pose_validity)} -> Frame 0: {camera_pose_validity[0]}")
            else:
                print(f"\nCamera Pose Validity: {camera_pose_validity.shape} -> Frame 0: {camera_pose_validity[0]}")
                

if __name__ == "__main__":
    # Path to the .pkl file
    pkl_file_path = 'sequenceFiles/sequenceFiles/test/downtown_rampAndStairs_00.pkl'

    # Create an instance of the 3DPWdataloader
    dataloader = ThreeDPWDataLoader(pkl_file_path)

    # Print out the information using the print_info method
    dataloader.print_info()
