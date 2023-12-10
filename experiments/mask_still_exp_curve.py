from pathlib import Path
FILE_DIR = Path(__file__).parent
import os,sys
import numpy as np, scipy.stats as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import pdb
sys.path.append(f'{FILE_DIR}/../depthai_blazepose')
import mediapipe_utils as mpu
sys.path.append(f'{FILE_DIR}/../traj_intention')
from Filter import Smooth_Filter

os.environ["DISPLAY"]=":0"
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

def compute_mean_and_conf_interval(accuracies, confidence=.95):
    accuracies = np.array(accuracies)
    n = len(accuracies)
    m, se = np.mean(accuracies), st.sem(accuracies)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

if __name__ == "__main__":
    DATA_DIR = f'{FILE_DIR}/../human_traj'
    data = {}
    for method in ["mask","nomask"]:
        data[method] = {}
        for human in ["abu","peiqi"]:
            data[method][human] = {}
            for index in ['001','002']:
                pkl_path = f'{DATA_DIR}/{human}_{method}_still/{human}_{method}_still{index}.pkl'
                pkl = open(pkl_path,'rb')
                pkl_file = pickle.load(pkl)
                poses = []
                for body in pkl_file:
                    poses.append(body.xyz)
                npy_file = np.array(poses)
                npy_file = 2*(npy_file-npy_file.min())/(npy_file.max()-npy_file.min())
                data[method][human][index] = npy_file

    data["mask"]['abu']['001'] = data["mask"]['abu']['001'][15:788] - data["mask"]['abu']['001'][15]
    data["mask"]['abu']['002'] = data["mask"]['abu']['002'][0:499] - data["mask"]['abu']['002'][0]
    data["mask"]["peiqi"]["001"] = data["mask"]["peiqi"]["001"][24:718] - data["mask"]["peiqi"]["001"][24]
    data["mask"]["peiqi"]["002"] = data["mask"]["peiqi"]["002"][0:912] - data["mask"]["peiqi"]["002"][0]

    data["nomask"]['abu']['001'] = data["nomask"]['abu']['001'][1:531] - data["nomask"]['abu']['001'][1]
    data["nomask"]['abu']['002'] = data["nomask"]['abu']['002'][0:985] - data["nomask"]['abu']['002'][66]
    data["nomask"]["peiqi"]["001"] = data["nomask"]["peiqi"]["001"][0:658] - data["nomask"]["peiqi"]["001"][0]
    data["nomask"]["peiqi"]["002"] = data["nomask"]["peiqi"]["002"][0:659] - data["nomask"]["peiqi"]["002"][73]

    # pdb.set_trace()
    filter_type = "kalman"
    smooth_filter = Smooth_Filter(filter_type)
    filtered_data = smooth_filter.smooth_trajectory(data["nomask"]["peiqi"]["001"])
    start = 320
    end = 450
    with plt.style.context(['ieee','notebook']):
        mean,conf = compute_mean_and_conf_interval(data["mask"]["peiqi"]["001"][:,0][start:end])
        plt.fill_between(np.arange(start,end),data["mask"]["peiqi"]["001"][:,0][start:end]-conf,data["mask"]["peiqi"]["001"][:,0][start:end]+conf,alpha=0.2)
        plt.plot(np.arange(start,end),data["mask"]["peiqi"]["001"][:,0][start:end],label="mask: variance {:.3f}".format(conf))
        mean,conf = compute_mean_and_conf_interval(data["nomask"]["peiqi"]["001"][:,0][start:end])
        plt.fill_between(np.arange(start,end),data["nomask"]["peiqi"]["001"][:,0][start:end]-conf,data["nomask"]["peiqi"]["001"][:,0][start:end]+conf,alpha=0.2)
        plt.plot(np.arange(start,end),data["nomask"]["peiqi"]["001"][:,0][start:end],label="no_mask: variance {:.3f}".format(conf),linestyle="solid")
        mean,conf = compute_mean_and_conf_interval(filtered_data[:,0][start:end])
        plt.fill_between(np.arange(start,end),filtered_data[:,0][start:end]-conf,filtered_data[:,0][start:end]+conf,alpha=0.2)
        plt.plot(np.arange(start,end),filtered_data[:,0][start:end],label="{:} filter: variance {:.3f}".format(filter_type,conf),linestyle="dashed")
        plt.plot(np.arange(start,end),np.zeros(658)[start:end],label="ground truth",linestyle="dotted")

        plt.xlabel('Frame')
        plt.ylabel('Keypoint Waving Loss')
        plt.title('Mask Ablation Study--Keypoint Waving Loss Comparison')
        plt.legend(loc='lower left',fontsize=10)
        plt.savefig(f'{FILE_DIR}/peiqi_001_mask_ablation.jpg')




