import argparse
import os
import torch
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import numpy as np
import cv2
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import pdb

from DLinear import Model
from Dataset import MyDataset


if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    JSON_FILE = f'{ROOT_DIR}/cut_traj.json'
    with open(JSON_FILE, 'r') as file:
        data_cut_points = json.load(file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_window', default=5,
                        help="looking back window size")
    parser.add_argument('--class_num', default=5,
                        help="number of classification categories")
    parser.add_argument('--individual', default=False, 
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--channels', type=int, default=15*3, 
                        help='encoder input size, channel here is set as upper body points * 3\
                        DLinear with --individual, use this hyperparameter as the number of channels') 
    parser.add_argument('--half_body', type=int, default=False, 
                        help='whether to extract only half body keypoints') 
    parser.add_argument('--epochs', type=int, default=40) 
    args = parser.parse_args()

    if args.half_body:
        args.channels = 10*3
    model = Model(args)
    checkpoint = torch.load(f'{FILE_DIR}/checkpoints/trail{args.epochs}.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = MyDataset(JSON_FILE,ROOT_DIR,args,type="train")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss(weight=dataset.weights)
    count = 0
    losses = 0
    error = [0]*5
    for batch in tqdm(dataloader):
        inputs, labels = batch
        outputs = model(inputs)
        gap = 1 if sum(abs(torch.argmax(outputs,1)-labels))>0 else 0
        count += gap
        if gap == 2:
            error[labels[0]] += 1
            error[labels[1]] += 1
        elif gap == 1:
            if labels[0] != torch.argmax(outputs,1)[0]:
                error[labels[0]] += 1
            else:
                error[labels[1]] += 1
        loss = criterion(outputs, labels)
        losses += loss.item()
    count = count / len(dataset)
    print(f"length of dataset:{len(dataset)}")
    print("accuracy: {:.2f}%".format((1 - count) * 100))
    print(f"loss:{losses/len(dataset)}")
    print(dataset.weights)
    error = [error[i]*dataset.weights[i] for i in range(len(dataset.weights))]
    print(error)
            


    

