import cv2
import os
import pdb

images_list = ['./close_estimation/images001','./human_traj/exp_hierarchical/images001']
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 15

for images_path in images_list:
    image = cv2.imread(os.path.join(images_path,os.listdir(images_path)[0]))
    img_w,img_h = image.shape[1],image.shape[0]
    # pdb.set_trace()
    video_path = os.path.join('/'.join(images_path.split('/')[:-1]),f"{images_path.split('/')[-1]}.mp4")
    video_out = cv2.VideoWriter(video_path, fourcc, fps, (img_w,img_h))
    print(video_path)
    images_sorted = os.listdir(images_path)
    images_sorted.sort(key=lambda x:int(x[:-4]))
    for image_name in images_sorted:
        image_path = os.path.join(images_path,image_name)
        image = cv2.imread(image_path)
        video_out.write(image)
    video_out.release()
