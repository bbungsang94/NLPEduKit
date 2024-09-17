import os
import numpy as np
import torch

DATA_ROOT = "D:\Creadto\Heritage\Dataset\Motion-X"

def main():
    root = DATA_ROOT
    motion = np.load(os.path.join(root, "motion_data/smplx_322/idea400/subset_0040/Act_Cute_During_Sitting.npy"))
    motion = torch.tensor(motion).float()
    motion_params = {
            'root_orient': motion[:, :3],  # controls the global root orientation
            'pose_body': motion[:, 3:3+63],  # controls the body
            'pose_hand': motion[:, 66:66+90],  # controls the finger articulation
            'pose_jaw': motion[:, 66+90:66+93],  # controls the yaw pose
            'face_expr': motion[:, 159:159+50],  # controls the face expression
            'face_shape': motion[:, 209:209+100],  # controls the face shape
            'trans': motion[:, 309:309+3],  # controls the global body position
            'betas': motion[:, 312:],  # controls the body shape. Body shape is static
        }

    # read text labels
    semantic_text = np.loadtxt(os.path.join(root, 'texts/semantic_labels/idea400/Act_cute_during_sitting_1_clip1.txt'),
                               dtype=str)
    pass

if __name__ == "__main__":
    main()