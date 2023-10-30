import argparse
import os
import torch
from pathlib import Path
FILE_DIR = Path(__file__).resolve().parent
import numpy as np
import cv2
import pdb

from DLinear import Model

class Args:
    def __init__(self,**kwargs):
        self.default = {"frame_window":5,"class_num":5,"individual":False,"channels":15*3,"half_body":False,"epochs":80}
        for key in ('frame_window', 'class_num', 'individual', 'channels','half_body','epochs'):
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
            checkpoint = torch.load(f'{FILE_DIR}/checkpoints/trail{args.epochs}.pth')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self,poses):
        outputs = self.model(poses)
        return outputs

if __name__ == '__main__':
    predictor = IntentionPredictor()
    pdb.set_trace()
        
