import os
import cv2
import time
import argparse

import torch
import model.detector
import utils.utils
from tqdm import tqdm
import pdb

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.data', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='modelzoo/coco2017-0.241078ap-model.pth', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--vid', type=str, default='', 
                        help='The path of test video')
    parser.add_argument('--out', type=str, default='', 
                        help='The path of out video')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"
    assert os.path.exists(opt.vid), "请指定正确的测试图像路径"
    if not opt.out:
        vid_name = os.path.split(opt.vid)[-1].split('.')[-2]
        opt.out = os.path.join('./out_vid',f'{vid_name}_result.mp4')

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    #sets the module in eval node
    model.eval()
    
    # 读取视频
    vid = cv2.VideoCapture(opt.vid)
    video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # 写入视频设置
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = vid.get(cv2.CAP_PROP_FPS)# 设置视频帧频
    size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 设置视频大小
    out = cv2.VideoWriter(opt.out,fourcc ,fps, size)

    #加载label names
    LABEL_NAMES = []
    with open(cfg["names"], 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    for i in tqdm(range(video_length)):
        ret, frame = vid.read()
        if not ret:
            break

        #数据预处理
        ori_img = frame
        res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0,3, 1, 2))
        img = img.to(device).float() / 255.0

        #模型推理
        # start = time.perf_counter()
        preds = model(img)
        # end = time.perf_counter()
        # forward_time = (end - start) * 1000.
        # print("forward time:%fms"%forward_time)

        #特征图后处理
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)

        h, w, _ = ori_img.shape
        scale_h, scale_w = h / cfg["height"], w / cfg["width"]

        #绘制预测框
        for box in output_boxes[0]:
            box = box.tolist()
        
            obj_score = box[4]
            category = LABEL_NAMES[int(box[5])]

            if category == 'person' and obj_score>0.2:
                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
                cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

        out.write(ori_img)
    

