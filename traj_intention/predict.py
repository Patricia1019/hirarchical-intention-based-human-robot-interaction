import argparse
import os
import torch
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import numpy as np
import cv2
import pdb

from DLinear import Model
from Dataset import INTENTION_LIST

class Args:
    def __init__(self,**kwargs):
        self.default = {"seq_len":5,"pred_len":5,"class_num":4,"individual":False,"channels":15*3,"half_body":False,"epochs":40}
        for key in ('seq_len','pred_len', 'class_num', 'individual', 'channels','half_body','epochs'):
            if key in kwargs and\
                kwargs[key]:
                    setattr(self, key, kwargs[key])
            else:
                setattr(self, key,self.default[key])
        

class IntentionPredictor:
    def __init__(self,ckpt_path=None,**kwargs): 
        args = Args()
        self.model = Model(args)
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
        else: # default
            checkpoint = torch.load(f'{FILE_DIR}/checkpoints/seq{args.seq_len}_pred{args.pred_len}_epoch{args.epochs}_whole.pth')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self,poses,restrict):
        pred_traj,pred_intention = self.model(poses)
        if pred_intention != INTENTION_LIST["no action"]:
            if restrict == "working_area":
                if pred_intention == INTENTION_LIST["get connectors"]:
                    pass # TODO
            else:
                intention = pred_intention
        else:
            intention = pred_intention
        return pred_traj,intention

if __name__ == '__main__':
    predictor = IntentionPredictor()
    pdb.set_trace()
        
