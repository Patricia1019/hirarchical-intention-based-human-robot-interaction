from pathlib import Path
FILE_DIR = Path(__file__).parent
import os,sys
import numpy as np
import pickle
import pandas as pd
import pdb
sys.path.append(f'{FILE_DIR}/../depthai_blazepose')
import mediapipe_utils as mpu


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
                    poses.append(body.landmarks)
                npy_file = np.array(poses)
                npy_file = 2*(npy_file-npy_file.min())/(npy_file.max()-npy_file.min())
                npy_file = camera_to_world(npy_file)
                npy_file[:, :, 2] -= np.min(npy_file[:, :, 2])
                data[method][human][index] = npy_file

    data["mask"]['abu']['001'] = data["mask"]['abu']['001'][15:788] - data["mask"]['abu']['001'][15]
    data["mask"]['abu']['002'] = data["mask"]['abu']['002'][0:499] - data["mask"]['abu']['002'][0]
    data["mask"]["peiqi"]["001"] = data["mask"]["peiqi"]["001"][24:718] - data["mask"]["peiqi"]["001"][24]
    data["mask"]["peiqi"]["002"] = data["mask"]["peiqi"]["002"][0:912] - data["mask"]["peiqi"]["002"][0]

    data["nomask"]['abu']['001'] = data["nomask"]['abu']['001'][1:531] - data["nomask"]['abu']['001'][1]
    data["nomask"]['abu']['002'] = data["nomask"]['abu']['002'][0:985] - data["nomask"]['abu']['002'][66]
    data["nomask"]["peiqi"]["001"] = data["nomask"]["peiqi"]["001"][0:658] - data["nomask"]["peiqi"]["001"][0]
    data["nomask"]["peiqi"]["002"] = data["nomask"]["peiqi"]["002"][0:659] - data["nomask"]["peiqi"]["002"][73]
    metric = {} # mask:2878; nomask:2832
    for method in ["mask","nomask"]:
        metric[method] = [0]
        for human in ["abu","peiqi"]:
            for index in ['001','002']:
                metric[method][0] += abs(data[method][human][index]).mean()
                # metric[method] += abs(data[method][human][index][1:] - data[method][human][index][:-1]).mean()
                # pdb.set_trace()
        metric[method][0] /= 4
        metric[method][0] = 1 - metric[method][0]
    
    print(metric)
    # df = pd.DataFrame(metric)

    # # 指定要保存的文件名
    # excel_file_path = f'{FILE_DIR}/metric.xlsx'

    # # 将数据框保存为 Excel 文件
    # df.to_excel(excel_file_path, index=False)

    # print(f'Data saved to {excel_file_path}')



