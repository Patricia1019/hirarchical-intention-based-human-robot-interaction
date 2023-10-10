import os,sys
import time
sys.path.append('../depthai_blazepose')
from BlazeposeDepthaiEdge import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
import pdb

if __name__ == '__main__':
    # 设置摄像头
    xyz = True
    internal_fps = 30
    internal_frame_height = 640
    no_pos_estimate = False
    tracker = BlazeposeDepthai(input_src="rgb", 
            pd_model=None,
            lm_model=None,
            smoothing=True,   
            xyz=xyz,            
            crop=False,
            internal_fps=internal_fps,
            internal_frame_height=internal_frame_height,
            force_detection=False,
            stats=True,
            trace=False,
            no_pos_estimate=no_pos_estimate)
    
    # 设置渲染器
    show_3d = False
    renderer = BlazeposeRenderer(
                tracker, 
                show_3d=show_3d, 
                output=None)
    
    count = 0
    t1 = time.time()
    while True:
        frame, body = tracker.next_frame()
        if frame is None: 
            print("frame is none!")
            break
        count += 1
        if count % 50 == 0:
            print(f"FPS:{count/(time.time()-t1)}")
        # 数据预处理
        frame = renderer.draw(frame, body)
        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            break