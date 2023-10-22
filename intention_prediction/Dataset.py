import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import argparse
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import pdb

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, json_file, root_dir, args, transform=None):
        self.json_file = json_file
        self.transform = transform
        self.root_dir = root_dir
        self.frame_window = args.frame_window
        self.channels = args.channels
        self.data = self.process_json()

    def process_json(self):
        with open(self.json_file, 'r') as file:
            data_cut_points = json.load(file)
        data = []
        for task in data_cut_points.keys():
            points = data_cut_points[task]
            npy_file = np.load(f'{self.root_dir}/{task[:-3]}/{task}.npy')
            npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)
            assert len(points['start']) == len(points['end']), f"The number of start points and end points doesn't match in {task}!"
            for i in range(len(points['end'])):
                end_index = points['end'][i] - 1
                posInputs1 = torch.from_numpy(npy_file[end_index-self.frame_window+1:end_index+1].reshape(self.frame_window,self.channels)).float()
                posInputs2 = torch.from_numpy(npy_file[end_index-self.frame_window:end_index].reshape(self.frame_window,self.channels)).float()
                posInputs3 = torch.from_numpy(npy_file[end_index-self.frame_window-1:end_index-1].reshape(self.frame_window,self.channels)).float()
                data.extend([(posInputs1,torch.tensor(1)),(posInputs2,torch.tensor(1)),(posInputs3,torch.tensor(1))])
                
                if i < len(points['end'])-1:
                    array = np.arange(end_index-self.frame_window//2,points['start'][i+1]-self.frame_window//2)
                else:
                    array = np.array([end_index,end_index+1,end_index+2])
                random_numbers = np.random.choice(array, 3, replace=False)
                for j in random_numbers:
                    if j+self.frame_window < npy_file.shape[0]-1:
                        negInput = torch.from_numpy(npy_file[j:j+self.frame_window].reshape(self.frame_window,self.channels)).float()
                    else:
                        negInput = torch.from_numpy(npy_file[j:npy_file.shape[0]].reshape(self.frame_window,self.channels).extend([0]*(j+self.frame_window-npy_file.shape[0]))).float()
                    data.append((negInput,torch.tensor(0)))
        
        return data
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    JSON_FILE = f'{ROOT_DIR}/cut_traj.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_window', default=10,
                        help="looking back window size")
    parser.add_argument('--class_num', default=2,
                        help="number of classification categories")
    parser.add_argument('--individual', default=False, 
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--channels', type=int, default=15*3, 
                        help='encoder input size, channel here is set as upper body points * 3\
                        DLinear with --individual, use this hyperparameter as the number of channels') 
    args = parser.parse_args()

    # 创建自定义数据集实例
    dataset = MyDataset(JSON_FILE,ROOT_DIR,args)

    # 创建数据加载器
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 遍历数据加载器以获取批次数据
    for batch in dataloader:
        inputs, labels = batch