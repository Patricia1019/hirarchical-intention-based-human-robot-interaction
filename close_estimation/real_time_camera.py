import os,sys
import time
from pathlib import Path
FILE_DIR = Path(__file__).parent
sys.path.append(f'{FILE_DIR}/../depthai_blazepose')
from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
import cv2
import pdb
import numpy as np
import argparse
import torch
import pickle
import shutil
sys.path.append(f'{FILE_DIR}/../traj_intention')
from predict import IntentionPredictor
from Dataset import INTENTION_LIST

def camera_to_world(X, R= np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32),\
                     t=0):
    return qrot(np.tile(R, (*X.shape[:-1], 1)), X) + t

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by 四元数quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = np.cross(qvec, v, len(q.shape) - 1)
    uuv = np.cross(qvec, uv, len(q.shape) - 1)
    return (v + 2 * (q[..., :1] * uv + uuv))

def get_intention(index):
    for key,value in INTENTION_LIST.items():
        if value == index:
            return key
    return "no_action"


if __name__ == '__main__':
    parentDir = Path(__file__).parent
    detection_nnPath = str((parentDir / Path('./models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())

    parser = argparse.ArgumentParser()
    parser.add_argument('--syncNN', action="store_true",
                    help="sync the output of detection network with the input of pose estimation network")
    parser.add_argument('--show', action="store_true",
                    help="show real time camera video")
    parser.add_argument('--task', default="test001",
                    help="name for traj.pkl and camera video")
    parser.add_argument('--seq_len', default=5,
                        help="input frame window")
    parser.add_argument('--send_window', default=10,
                        help="send if intention is consecutively recognized in send_window")
    parser.add_argument('--video', action="store_true",
                    help="save video, else save images")
    parser.add_argument('--restrict', type=str,default="ood",
                        help='four options:[no,working_area,ood,all]') 
    parser.add_argument('--outer_restrict',type=str,default="working_area",
                        help='outer restriction')
    args = parser.parse_args()


    # 设置摄像头
    xyz = True
    internal_fps = 30
    internal_frame_height = 300
    no_pos_estimate = False
    tracker = BlazeposeDepthai(input_src="rgb", 
            pd_model=None,
            lm_model=None,
            smoothing=True,   
            xyz=xyz,            
            crop=False,
            internal_fps=internal_fps,
            internal_frame_height=internal_frame_height,
            force_detection=False,
            stats=True,
            trace=False,
            no_pos_estimate=no_pos_estimate)
    
    # 设置渲染器
    show_3d = False
    renderer = BlazeposeRenderer(
                tracker, 
                show_3d=show_3d, 
                output=None)
    
    count = 0
    t1 = time.time()
    traj_queue = []
    frame_size = 5
    Predictor = IntentionPredictor()
    send_window = 10
    traj = []
    video = False
    task = args.task
    ROOT_DIR = f'{FILE_DIR}/../human_traj/{task[:-3]}'
    if not os.path.exists(ROOT_DIR):
        os.mkdir(ROOT_DIR)
    if os.path.exists(f'{ROOT_DIR}/images{task[-3:]}'):
        # If it exists, remove the entire directory and its contents
        shutil.rmtree(f'{ROOT_DIR}/images{task[-3:]}')
    os.makedirs(f'{ROOT_DIR}/images{task[-3:]}')
    while True:
        frame, body = tracker.next_frame()
        if frame is None: 
            print("frame is none!")
            break
        if body:
            upperbody = np.concatenate((body.landmarks[11:25,:],body.landmarks[0:1,:]),axis=0)
            body.landmarks = upperbody
            frame = renderer.draw(frame, body)
            key = renderer.waitKey(delay=1)
        if body:
            landmarks = body.landmarks_world
            righthand = landmarks[16]
            rightelbow = landmarks[14]
            rightshoulder = landmarks[12]
            # upperbody = np.concatenate((landmarks[11:25,:],landmarks[0:1,:]),axis=0)
            upperbody = body.landmarks
            if len(traj_queue) < frame_size:
                traj_queue.append(upperbody)
            else:
                traj_queue.pop(0)
                traj_queue.append(upperbody)
            assert len(traj_queue) <= frame_size, "the length of accumulated traj is longer than intention prediction frame size!"
            # intention prediction based on learning 
            intention = None
            if len(traj_queue)==frame_size: # send to intention prediction module
                poses = np.array(traj_queue)
                poses_norm = 2*(poses-poses.min())/(poses.max()-poses.min())
                poses_world = camera_to_world(poses_norm)
                poses_world[:, :, 2] -= np.min(poses_world[:, :, 2])
                inputs = torch.tensor(poses_world.reshape(1,frame_size,-1)).float()
                # pdb.set_trace()
                pred_traj,pred_intention = Predictor.predict(inputs,'ood')
                intention = get_intention(pred_intention)
                cv2.putText(frame, f"intention:{intention}", (2, frame.shape[0] - 52), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))


            traj.append(body)
            if not video:
                if not os.path.exists(f'{ROOT_DIR}/images{task[-3:]}'):
                    os.mkdir(f'{ROOT_DIR}/images{task[-3:]}')
                cv2.imwrite(f'{ROOT_DIR}/images{task[-3:]}/{count}.png',frame)
                    
            count += 1
            if count % 50 == 0:
                print(f"FPS:{count/(time.time()-t1)}")

        # cv2.imwrite(f'./images/{count}.png',frame)
        if cv2.waitKey(1) == ord('q'):
            break

    file = open(f'{ROOT_DIR}/{task}.pkl', 'wb')
    pickle.dump(traj, file)