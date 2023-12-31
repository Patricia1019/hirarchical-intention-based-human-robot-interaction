import numpy as np
import pickle
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import cv2
import os,sys
sys.path.append('./depthai_blazepose')
from o3d_utils import Visu3D
import mediapipe_utils as mpu

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--task', default="test",
                    help="name for traj.pkl and camera video")
parser.add_argument('--input_type', default="images",
                    help="type for inputs")
args = parser.parse_args()
task = args.task
ROOT_DIR = f'{FILE_DIR}/{task[:-3]}'
os.environ["DISPLAY"]=":0"
class Skeleton:
    def parents(self):
        # parents = [0,0,11,12,13,14,15,16,15,16,15,16,11,12]
        parents = [14,14,0,1,2,3,4,5,4,5,4,5,0,1]
        # others = [24,19,20]
        others = [13,8,9]
        parents.extend(others)
        return np.array(parents)

    def children(self):
        # children = np.arange(11,25)
        children = np.arange(0,14)
        # others = np.array([23,17,18])
        others = np.array([12,6,7])
        children = np.append(children,others)
        return children

    def joints_right(self):
        right = []
        # for i in range(12,26,2):
        for i in range(1,15,2):
            right.append(i)
        return right

    def joints_thumb(self):
        return [21,22]
        # return [10,11]

def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)

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


def camera_to_world(X, R= np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32),\
                     t=0):
    return qrot(np.tile(R, (*X.shape[:-1], 1)), X) + t


def read_video(filename, fps=None, skip=0, limit=-1):
    stream = cv2.VideoCapture(filename)

    i = 0
    while True:
        grabbed, frame = stream.read()
        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            print('===========================> This video get ' + str(i) + ' frames in total.')
            sys.stdout.flush()
            break

        i += 1
        if i > skip:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield np.array(frame)
        if i == limit:
            break

def render(poses, output, skeleton=Skeleton(), fps=6, bitrate=30000, azim=np.array(70., dtype=np.float32), \
           viewport=(1000, 1002),limit=-1, downsample=1, size=5, task=None, input_video_skip=0,input_type=None):
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input', pad=60, fontsize=15)

    # prevent wired error
    _ = Axes3D.__class__.__name__

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        # ax.dist = 2
        ax.set_title(title, pad=0, fontsize=15)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video or images
    all_frames = []
    if input_type == "video":
        input_video_path = f'{ROOT_DIR}/{task}_camera_out.mp4'
        for f in read_video(input_video_path, fps=None, skip=input_video_skip):
            all_frames.append(f)
    elif input_type == "images":
        input_image_path = f'{ROOT_DIR}/images{task[-3:]}'
        images = os.listdir(input_image_path)
        images.sort(key=lambda x:int(x[:-4]))
        for img in images:
            img = cv2.imread(f'{input_image_path}/{img}',cv2.IMREAD_COLOR)
            b,g,r = cv2.split(img)
            img = cv2.merge((r,g,b))
            all_frames.append(img)
    else:
        print("Invalid Input! Using black background!")
        # Black background
        all_frames = np.zeros((poses[0].shape[0], viewport[1], viewport[0]), dtype='uint8')


    if downsample > 1:
        # keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    children = skeleton.children()
    pbar = tqdm(total=limit)


    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                j_children = children[j]
                if j_parent == -1:
                    continue

                col = 'red' if j_children in skeleton.joints_right() else 'black'
                col = 'green' if j_children in skeleton.joints_thumb() else col
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j_children, 0], pos[j_parent, 0]],
                                               [pos[j_children, 1], pos[j_parent, 1]],
                                               [pos[j_children, 2], pos[j_parent, 2]], zdir='z', c=col))

            # points = ax_in.scatter(*keypoints[i].T, 5, color='red', edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                j_children = children[j]
                if j_parent == -1:
                    continue

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j][0].set_xdata(np.array([pos[j_children, 0], pos[j_parent, 0]])) # Hotfix matplotlib's bug. https://github.com/matplotlib/matplotlib/pull/20555
                    lines_3d[n][j][0].set_ydata([pos[j_children, 1], pos[j_parent, 1]])
                    lines_3d[n][j][0].set_3d_properties([pos[j_children, 2], pos[j_parent, 2]], zdir='z')

            # points.set_offsets(keypoints[i])

        pbar.update()

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=limit, interval=1000.0 / fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=60, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    pbar.close()
    plt.close()

# ROOT_DIR = f'{FILE_DIR}/../close_estimation/{task[:-3]}'
pic = open(f'{ROOT_DIR}/{task}.pkl','rb')
traj = pickle.load(pic)
# pdb.set_trace()
# trajreader = BlazeposeTrajRenderer(show_3d='world')
poses = []
for body in traj:
    poses.append(body.landmarks)
poses = np.array(poses)
poses_norm = 3*(poses-poses.min())/(poses.max()-poses.min())
# poses_norm = 2*(poses_norm-poses_norm.min(0).min(0))/(poses_norm.max(0).max(0)-poses_norm.min(0).min(0))
poses_world = camera_to_world(poses_norm)
poses_world[:, :, 2] -= np.min(poses_world[:, :, 2])
# pdb.set_trace()
# np.save(f'{ROOT_DIR}/{task}.npy',poses_world)
predictions = {"Pose":poses_world}
# ROOT_DIR = f'{FILE_DIR}/../close_estimation/{task[:-3]}'
render(predictions,f'{ROOT_DIR}/{task}.mp4',task=task,input_type=args.input_type)
print("ok")
