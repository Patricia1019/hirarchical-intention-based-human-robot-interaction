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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import pdb

from DLinear import Model_FinalIntention,Model_FinalTraj
from Dataset import MyDataset, INTENTION_LIST
from predict import IntentionPredictor

os.environ["DISPLAY"]=":0"
if __name__ == '__main__':
    ROOT_DIR = f'{FILE_DIR}/../human_traj'
    
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
    parser.add_argument('--test_whole', default=True,
                        help='whether to test on build_cars tasks')
    parser.add_argument('--no_mask', action="store_true",
                        help='whether to remove mask')
    parser.add_argument('--restrict', type=str,default="ood",
                        help='four options:[no,working_area,ood,all]')
    parser.add_argument('--test_type',type=str,default="test_self",
                        help='two options:[test_self,test_others]')
    parser.add_argument('--input_type', type=str,default="pkl",
                        help='two options:[pkl,npy]') 
    parser.add_argument('--model_type',type=str,default="final_traj",
                        help='two options:[final_intention,final_traj]')
    parser.add_argument('--epochs', type=int, default=40) 
    args = parser.parse_args()

    if args.half_body:
        args.channels = 10*3
    
    if args.no_mask:
        JSON_FILE = f'{ROOT_DIR}/cut_traj_nomask.json'
    else:
        JSON_FILE = f'{ROOT_DIR}/cut_traj_mask.json'

    ckpt_path = f'{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{args.epochs}_whole_{args.input_type}_{args.model_type}_nomask{args.no_mask}.pth'
    dataset = MyDataset(JSON_FILE,ROOT_DIR,args,dataset_type=args.test_type,test_whole=args.test_whole,input_type=args.input_type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # pdb.set_trace()
    count = 0
    losses = 0
    error = [0]*args.class_num
    intention_list = []
    labels_list = []
    predictor = IntentionPredictor(model_type=args.model_type,no_mask=args.no_mask)
    for batch in tqdm(dataloader):
        inputs,target_traj,labels = batch # inputs.shape:[batch_size,seq_len,15*3]
        # pred_traj,pred_intention = model(inputs)
        pred_traj,pred_intention = predictor.predict(inputs,args.restrict)
        gap = 1 if sum(abs(pred_intention-labels))>0 else 0
        count += gap
        if gap == 1:
            error[labels[0]] += 1
        # pdb.set_trace()
        intention_list.append(pred_intention[0].item())
        labels_list.append(labels[0].item())

    intention_list = np.array(intention_list)
    labels_list = np.array(labels_list)
    confusion_matrix = metrics.confusion_matrix(labels_list,intention_list)
    confusion_matrix_norm = metrics.confusion_matrix(labels_list,intention_list,normalize='true')
    # pdb.set_trace()
    # confusion_matrix_norm = confusion_matrix/confusion_matrix.sum(1)
    cm_display_norm = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_norm, display_labels = ["no_action","connectors","screws","wheels"])
    cm_display_norm.plot()
    if args.test_whole:
        plt.savefig(f'{FILE_DIR}/results/cm_norm_test_whole_{args.restrict}_restrict_{args.test_type}_{args.model_type}_nomask{args.no_mask}.jpg', bbox_inches = 'tight')
    else:
        plt.savefig(f'{FILE_DIR}/results/cm_norm_not_test_whole_{args.restrict}_restrict_{args.test_type}_{args.model_type}_nomask{args.no_mask}.jpg', bbox_inches = 'tight')
    plt.close()
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["no_action","connectors","screws","wheels"])
    cm_display.plot()
    # if args.test_whole:
    #     plt.savefig(f'{FILE_DIR}/results/cm_test_whole_{args.restrict}_restrict_{args.test_type}_{args.model_type}_nomask{args.no_mask}.jpg', bbox_inches = 'tight')
    # else:
    #     plt.savefig(f'{FILE_DIR}/results/cm_not_test_whole_{args.restrict}_restrict_{args.test_type}_{args.model_type}_nomask{args.no_mask}.jpg', bbox_inches = 'tight')
    count = count / len(dataset)
    print(f"length of dataset:{len(dataset)}")
    print(f"accuracy: {1 - count}")
    print(f"loss:{losses/len(dataset)}")
    print("dataset.weights")
    print(dataset.weights)
    print("error:")
    error = [error[i]*dataset.weights[i] for i in range(len(dataset.weights))]
    for key,value in INTENTION_LIST.items():
        print(f"{key}: {1-error[value]}")
    # print(error)
            


    

