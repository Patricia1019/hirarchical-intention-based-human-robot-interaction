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

    net = Model(args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for task in data_cut_points.keys():
            points = data_cut_points[task]
            npy_file = np.load(f'{ROOT_DIR}/{task[:-3]}/{task}.npy')
            npy_file = np.concatenate((npy_file[:,11:25,:],npy_file[:,0:1,:]),axis=1)
            assert len(points['start']) == len(points['end']), f"The number of start points and end points doesn't match in {task}!"
            print(task)
            for i in tqdm(range(len(points['end']))):
                end_index = points['end'][i]
                inputs1 = torch.from_numpy(npy_file[end_index-args.frame_window+1:end_index+1].reshape(1,args.frame_window,args.channels)).float()
                labels1 = torch.tensor([1])
                inputs0 = torch.from_numpy(npy_file[end_index+1:end_index+11].reshape(1,args.frame_window,args.channels)).float()
                labels0 = torch.tensor([0])
                optimizer.zero_grad()
                outputs1 = net(inputs1)
                loss1 = criterion(outputs1, labels1)
                loss1.backward()
                optimizer.step()
                running_loss += loss1.item()

                optimizer.zero_grad()
                outputs0 = net(inputs0)
                loss0 = criterion(outputs0, labels0)
                loss0.backward()
                optimizer.step()
                running_loss += loss0.item()
                count += 2
            # for i in tqdm(range(10,npy_file.shape[0])):
            #     index = np.digitize(i, points['start']) # find i in what range of points['start']
            #     inputs = torch.from_numpy(npy_file[i-args.frame_window+1:i+1].reshape(1,args.frame_window,args.channels)).float()
            #     if points['end'][index-1] >= i and (i-points['start'][index-1]) >= (args.frame_window//2): # getting connectors
            #         labels = torch.tensor([1])
            #     else: # not getting connectors
            #         labels = torch.tensor([0])
            #     optimizer.zero_grad()
            #     outputs = net(inputs)
            #     loss = criterion(outputs, labels)
            #     loss.backward()
            #     optimizer.step()
            #     running_loss += loss.item()
            #     count += 1
                
        print(f'Epoch {epoch + 1}, Loss: {running_loss / count}')
    
    torch.save(net.state_dict(), f'{FILE_DIR}/checkpoints/trail{epochs}.pth')
    

