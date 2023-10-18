import numpy as np
import pickle
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import cv2
import os,sys
sys.path.append('./depthai_blazepose')
from o3d_utils import Visu3D
import mediapipe_utils as mpu
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="test",
                    help="name for traj.pkl and camera video")
args = parser.parse_args()
task = args.task
ROOT_DIR = f'{FILE_DIR}/{task[:-3]}'

cap = cv2.VideoCapture(f'{ROOT_DIR}/{task}_camera_out.mp4')
body_traj = np.load(f'{ROOT_DIR}/{task}.npy')
assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == body_traj.shape[0]

pdb.set_trace()