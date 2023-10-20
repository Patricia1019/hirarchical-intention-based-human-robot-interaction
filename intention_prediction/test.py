import argparse
import os
import torch
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import numpy as np
import cv2
from tqdm import tqdm
import json
import pdb

from DLinear import Model


if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    JSON_FILE = f'{ROOT_DIR}/cut_traj.json'
    with open(JSON_FILE, 'r') as file:
        data_cut_points = json.load(file)
    
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

    model = Model(args)
    checkpoint = torch.load(f'{FILE_DIR}/checkpoints/trail10.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    for task in data_cut_points.keys():
        points = data_cut_points[task]
        npy_file = np.load(f'{ROOT_DIR}/{task[:-3]}/{task}.npy')
        npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)
        print(task)
        count = 0
        for i in range(10,npy_file.shape[0]):
            index = np.digitize(i, points['start']) # find i in what range of points['start']
            inputs = torch.from_numpy(npy_file[i-args.frame_window+1:i+1].reshape(1,args.frame_window,args.channels)).float()
            if points['end'][index-1] >= i and (i-points['start'][index-1]) >= (args.frame_window//2): # getting connectors
                labels = torch.tensor([1])
            else: # not getting connectors
                labels = torch.tensor([0])
            count += abs(torch.argmax(model(inputs))-labels)/npy_file.shape[0]
        print(count)
            


    

