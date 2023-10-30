import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import argparse
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import pdb

INTENTION_LIST = {"get_connector":1,"stop":2,"straight_up":3,"wave_hand":4}
class MyDataset(Dataset):
    def __init__(self, json_file, root_dir, args,type="train",transform=None):
        self.json_file = json_file
        self.transform = transform
        self.root_dir = root_dir
        self.frame_window = args.frame_window
        self.channels = args.channels
        self.type = type
        self.half_body = args.half_body
        self.intention_list = INTENTION_LIST
        self.data,self.weights = self.process_json()

    def process_json(self):
        with open(self.json_file, 'r') as file:
            data_cut_points = json.load(file)
        data_cut_points = data_cut_points[self.type] # TODO
        data = []
        weights = [0]*(len(self.intention_list)+1)
        for intention in data_cut_points.keys():
            intention_tasks = data_cut_points[intention]
            intention_label = self.intention_list[intention]
            for task in intention_tasks.keys():
                points = intention_tasks[task]
                npy_file = np.load(f'{self.root_dir}/{task[:-3]}/{task}.npy')
                npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)
                if self.half_body:
                    npy_file = np.concatenate((npy_file[:,1::2,:],npy_file[:,0:1,:],npy_file[:,-1:,:],npy_file[:,-3:-2,:]),axis=1)
                assert len(points['start']) == len(points['end']), f"The number of start points and end points doesn't match in {task}!"
                task_data = []
                for i in range(len(points['end'])):
                    end_index = points['end'][i] - 1
                    start_index = points['start'][i] - 1
                    for j in range(start_index,end_index-self.frame_window+1):
                        posInputs = torch.from_numpy(npy_file[j:j+self.frame_window].reshape(self.frame_window,self.channels)).float()
                        task_data.append((posInputs,torch.tensor(intention_label)))
                        weights[intention_label] += 1
                    
                pos_data_num = len(task_data) // (len(self.intention_list)+1)
                array = []
                array.extend(np.arange(0,points['start'][0]-self.frame_window))
                for i in range(1,len(points['end'])-1):
                    end_index = points['end'][i]
                    start_index = points['start'][i+1]
                    array.extend(np.arange(end_index+self.frame_window//2,start_index-self.frame_window))
                array.extend(np.arange(points['end'][-1]+self.frame_window//2,npy_file.shape[0]-self.frame_window))
                try:
                    random_numbers = np.random.choice(array, pos_data_num, replace=False)
                except:
                    random_numbers = np.random.choice(array, pos_data_num, replace=True)
                for j in random_numbers:
                    negInputs = torch.from_numpy(npy_file[j:j+self.frame_window].reshape(self.frame_window,self.channels)).float()
                    task_data.append((negInputs,torch.tensor(0)))
                    weights[0] += 1


                data.extend(task_data)
        weights = torch.tensor([1/x for x in weights])
        return data,weights
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def get_weights(self):
        return self.weights

if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    JSON_FILE = f'{ROOT_DIR}/cut_traj.json'

    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_window', default=5,
                        help="looking back window size")
    parser.add_argument('--class_num', default=2,
                        help="number of classification categories")
    parser.add_argument('--individual', default=False, 
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--channels', type=int, default=15*3, 
                        help='encoder input size, channel here is set as upper body points * 3\
                        DLinear with --individual, use this hyperparameter as the number of channels') 
    parser.add_argument('--half_body', type=int, default=False, 
                        help='whether to extract only half body keypoints') 
    args = parser.parse_args()

    dataset = MyDataset(JSON_FILE,ROOT_DIR,args)

    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        inputs, labels = batch