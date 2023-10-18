import numpy as np
import pickle
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import cv2

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import pdb


class Skeleton:
    def parents(self):
        parents = [-1]*11
        others = [0,0,11,12,13,14,15,16,15,16,15,16,11,12,23,24,25,26,-1,-1,-1,-1]
        parents.extend(others)
        return np.array(parents)

    def joints_right(self):
        right = []
        for i in range(12,30,2):
            right.append(i)
        return right

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

def render(poses, output, skeleton=Skeleton(), fps=5, bitrate=30000, azim=np.array(70., dtype=np.float32), \
           viewport=(1000, 1002),limit=-1, downsample=1, size=5, input_video_path=None, input_video_skip=0):
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))
    # ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    # ax_in.get_xaxis().set_visible(False)
    # ax_in.get_yaxis().set_visible(False)
    # ax_in.set_axis_off()
    # ax_in.set_title('Input')

    # prevent wired error
    _ = Axes3D.__class__.__name__

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(1, 1 + len(poses), index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 12.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video_path is None:
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
    pbar = tqdm(total=limit)


    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        if not initialized:
            # image = ax_in.imshow(all_frames[i], aspect='equal')

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                # if len(parents) == keypoints.shape[1] and 1 == 2:
                #     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                #     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                #                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            # points = ax_in.scatter(*keypoints[i].T, 5, color='red', edgecolors='white', zorder=10)

            initialized = True
        else:
            # image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                # if len(parents) == keypoints.shape[1] and 1 == 2:
                #     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                #                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 11][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]])) # Hotfix matplotlib's bug. https://github.com/matplotlib/matplotlib/pull/20555
                    lines_3d[n][j - 11][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 11][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')

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


pic = open(f'{FILE_DIR}/test.pkl','rb')
traj = pickle.load(pic)
# trajreader = BlazeposeTrajRenderer(show_3d='world')
poses = []
for body in traj:
    poses.append(body.landmarks)
poses = np.array(poses)
poses = 2*(poses-poses.min())/(poses.max()-poses.min())-1
poses = camera_to_world(poses)
poses[:, :, 2] -= np.min(poses[:, :, 2])
predictions = {"ours":poses}
render(predictions,f'{FILE_DIR}/test.mp4')
print("ok")
