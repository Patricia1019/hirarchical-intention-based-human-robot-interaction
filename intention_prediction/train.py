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
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--epochs', type=int, default=40) 
    args = parser.parse_args()

    if args.half_body:
        args.channels = 10*3
    net = Model(args)

    batch_size = args.batch_size
    dataset = MyDataset(JSON_FILE,ROOT_DIR,args,type="train")
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

    criterion = torch.nn.CrossEntropyLoss(weight=dataset.weights)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochs = args.epochs
    for epoch in range(epochs):
        running_loss = 0.0
        count = 0
        for batch in tqdm(dataloader):
            inputs, labels = batch
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1
                
        print(f'Epoch {epoch + 1}, Loss: {running_loss / count}')
    
    torch.save(net.state_dict(), f'{FILE_DIR}/checkpoints/trail{epochs}.pth')
    

