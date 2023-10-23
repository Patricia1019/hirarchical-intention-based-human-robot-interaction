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
    parser.add_argument('--class_num', default=2,
                        help="number of classification categories")
    parser.add_argument('--individual', default=False, 
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--channels', type=int, default=15*3, 
                        help='encoder input size, channel here is set as upper body points * 3\
                        DLinear with --individual, use this hyperparameter as the number of channels') 
    parser.add_argument('--half_body', type=int, default=True, 
                        help='whether to extract only half body keypoints') 
    args = parser.parse_args()

    if args.half_body:
        args.channels = 10*3
    model = Model(args)
    checkpoint = torch.load(f'{FILE_DIR}/checkpoints/trail40.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = MyDataset(JSON_FILE,ROOT_DIR,args,type="test")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    count = 0
    for batch in tqdm(dataloader):
        inputs, labels = batch
        outputs = model(inputs)
        count += sum(abs(torch.argmax(outputs,1)-labels))
    count = count / len(dataloader)
    print(count)
            


    

