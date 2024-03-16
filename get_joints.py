import os
import glob
import json
import subprocess

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset_from_cfg, expand_source_paths

from humor.humor_model import HumorModel
from optim.base_scene import BaseSceneModel
from optim.moving_scene import MovingSceneModel
from optim.optimizers import (
    RootOptimizer,
    SmoothOptimizer,
    SMPLOptimizer,
    MotionOptimizer,
    MotionOptimizerChunks,
)
from optim.output import (
    save_track_info,
    save_camera_json,
    save_input_poses,
    save_initial_predictions,
)
from vis.viewer import init_viewer

from util.loaders import (
    load_vposer,
    load_state,
    load_gmm,
    load_smpl_body_model,
    resolve_cfg_paths,
)
from util.logger import Logger
from util.tensor import get_device, move_to, detach_all, to_torch

from run_vis import run_vis

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

from body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose, run_smpl

result_path = '/content/shorto_000060_world_results.npz'
smoothed_data = np.load(result_path)

print(smoothed_data.files)
device = torch.device("cpu")



def run_make_dummy_model():

    num_people = smoothed_data['trans'].shape[0]  # Assuming the first dimension is the number of people detected.
    results = []  # To store results for each person.

    path_new = '/content/slahmr/_DATA/body_models/smplh/neutral/model.npz'

    for i in range(num_people):
        human_T = smoothed_data['trans'][i,:,:]
        human_T = np.expand_dims(human_T, axis=0)
        human_T = torch.tensor(human_T, dtype=torch.float32, device=device)

        human_R = smoothed_data['root_orient'][i,:,:]
        human_R = np.expand_dims(human_R, axis=0)
        human_R = torch.tensor(human_R, dtype=torch.float32, device=device)

        body_pose = smoothed_data['pose_body'][i,:,:]
        body_pose = np.expand_dims(body_pose, axis=0)
        body_pose = torch.tensor(body_pose, dtype=torch.float32, device=device)

        betas = smoothed_data['betas'][i]
        betas = torch.tensor(betas, dtype=torch.float32).to(device)

        # Assuming B and T are batch size and sequence length respectively. Adjust as necessary.
        B = human_T.shape[0]
        T = human_T.shape[1]

        # Load model and run SMPL for each individual.
        body_model, fit_gender = load_smpl_body_model(path_new, B * T, device=device)
        smpl_Out = run_smpl(body_model, human_T, human_R, body_pose, betas)

        # Adjusting to dictionary access.
        joints = smpl_Out['joints'].detach().cpu().numpy()  # Assuming 'joints' is the correct key.
        vertices = smpl_Out['vertices'].detach().cpu().numpy()  # Assuming 'vertices' is the correct key.

        results.append({
            'joints': joints,
            'vertices': vertices
        })

    # Save results for all people to an NPZ file.
    np.savez('/content/joints_results.npz', *results)

    print("Complete")


def main():
  run_make_dummy_model()
    


if __name__ == "__main__":
    main()