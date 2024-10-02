import os
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

DATA_ROOT = "D:\Creadto\Heritage\Dataset\Motion-X"

def main():
    root = DATA_ROOT
    # npy_path = r"D:\Creadto\Heritage\Dataset\Motion-X\face_motion_data\smplx_322\humanml\000046_clip0000.npy"
    # motion = np.load(npy_path)
    motion = np.load(os.path.join(root, "motion_data/smplx_322/humman/subset_0000/A_Hero_S_Positive.npy"))
    motion = torch.tensor(motion).float()
    # read text labels
    semantic_text = np.loadtxt(os.path.join(root, 'texts/semantic_labels/idea400/Holding_chopsticks_in_hand_during_sitting_clip1.txt'),
                               dtype=str)
    make_motion_media(text=" ".join([x for x in semantic_text]),
                      motion=motion)

def make_motion_media(text, motion, model_path="./models/smplx/SMPLX_NEUTRAL_2020.npz", device="cuda:0"):
    from smplx import SMPLX    
    frames = motion.shape[0]
    smplx = SMPLX(model_path=model_path, use_pca=False, flat_hand_mean=False, num_expression_coeffs=50, use_hands=True).eval().to(device)
    motion = motion.to(device)            
    faces = smplx.faces
    
    for i in range(frames):
        motion_params = {
            'global_orient': motion[i, :3].unsqueeze(dim=0),  # controls the global root orientation
            'body_pose': motion[i, 3:3+63].unsqueeze(dim=0),  # controls the body
            'left_hand_pose': motion[i, 66:66+45].unsqueeze(dim=0),  # controls the finger articulation
            'right_hand_pose': motion[i, 111:111+45].unsqueeze(dim=0),  # controls the finger articulation
            'jaw_pose': motion[i, 156:156+3].unsqueeze(dim=0),  # controls the yaw pose
            'expression': motion[i, 159:159+50].unsqueeze(dim=0),  # controls the face expression
            # 'face_shape': motion[:, 209:209+100],  # controls the face shape
            'transl': motion[i, 309:309+3].unsqueeze(dim=0),  # controls the global body position
            'betas': motion[i, 312:].unsqueeze(dim=0),  # controls the body shape. Body shape is static
            'leye_pose': torch.zeros([1, 3]).to(device),
            'reye_pose': torch.zeros([1, 3]).to(device),
        }
        output = smplx.forward(**motion_params)
        vertices = output.vertices
        save_path = os.path.join("./debug/motion_sample", "%06d.jpg" % i)
        processed_frame = render_mesh_to_image(vertices[0].cpu().detach().numpy(), faces, save_path)
    pass

def render_mesh_to_image(vertices, faces, image_path):
    # Trimesh를 사용하여 mesh 객체 생성
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 3D 시각화를 위한 figure 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 삼각형 면 그리기
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, cmap='viridis')

    # 축을 없애고 보기 좋게 설정
    ax.set_axis_off()
    ax.view_init(elev=90, azim=270)  # 시야각도 설정 (필요에 따라 수정 가능)

    # 이미지를 파일로 저장
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    main()