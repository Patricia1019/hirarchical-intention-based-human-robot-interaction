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
from Dataset import MyDataset, INTENTION_LIST


if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    JSON_FILE = f'{ROOT_DIR}/cut_traj_new.json'
    with open(JSON_FILE, 'r') as file:
        data_cut_points = json.load(file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', default=5,
                        help="input frame window")
    parser.add_argument('--pred_len', default=5,
                        help="output predicted frame length")
    parser.add_argument('--class_num', default=4,
                        help="number of classification categories")
    parser.add_argument('--individual', default=False, 
                        help='DLinear: a linear layer for each variate(channel) individually')
    parser.add_argument('--channels', type=int, default=15*3, 
                        help='encoder input size, channel here is set as upper body points * 3\
                        DLinear with --individual, use this hyperparameter as the number of channels') 
    parser.add_argument('--half_body', type=int, default=False, 
                        help='whether to extract only half body keypoints') 
    parser.add_argument('--test_whole', action="store_true",
                        help='whether to test on build_cars tasks') 
    parser.add_argument('--epochs', type=int, default=40) 
    args = parser.parse_args()

    if args.half_body:
        args.channels = 10*3
    model = Model(args)
    checkpoint = torch.load(f'{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{args.epochs}.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = MyDataset(JSON_FILE,ROOT_DIR,args,type="test",test_whole=args.test_whole)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    class_criterion = torch.nn.CrossEntropyLoss(weight=dataset.weights)
    traj_criterion = torch.nn.MSELoss()
    count = 0
    losses = 0
    error = [0]*args.class_num
    for batch in tqdm(dataloader):
        inputs,target_traj,labels = batch
        pred_traj,pred_intention = model(inputs)
        gap = 1 if sum(abs(torch.argmax(pred_intention,1)-labels))>0 else 0
        count += gap
        if gap == 2:
            error[labels[0]] += 1
            error[labels[1]] += 1
        elif gap == 1:
            if labels[0] != torch.argmax(pred_intention,1)[0]:
                error[labels[0]] += 1
            else:
                error[labels[1]] += 1
        traj_loss = traj_criterion(pred_traj,target_traj)
        intention_loss = class_criterion(pred_intention, labels)
        loss = traj_loss + intention_loss
        losses += loss.item()
    count = count / len(dataset)
    print(f"length of dataset:{len(dataset)}")
    print("accuracy: {:.2f}%".format((1 - count) * 100))
    print(f"loss:{losses/len(dataset)}")
    print("dataset.weights")
    print(dataset.weights)
    print("error:")
    error = [error[i]*dataset.weights[i] for i in range(len(dataset.weights))]
    for key,value in INTENTION_LIST.items():
        print(f"{key}: {error[value].item()}")
    # print(error)
            


    

