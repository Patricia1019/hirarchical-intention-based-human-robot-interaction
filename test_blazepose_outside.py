import os,sys
import cv2
import numpy as np
import depthai as dai
from pathlib import Path
import pdb
import marshal
import time
import argparse
sys.path.append('./depthai_blazepose')
from BlazeposeRenderer import BlazeposeRenderer
from BlazeposeDepthaiEdge_module_outside import BlazeposeDepthaiModule

if __name__ == '__main__':
    pipeline = dai.Pipeline()
    # define sources
    camRGB = pipeline.createColorCamera()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)

    # define in/out nodes
    xoutRGB = pipeline.create(dai.node.XLinkOut)
    xoutRGB.setStreamName("RGB")
    xoutBlazepose = pipeline.create(dai.node.XLinkOut)
    xoutBlazepose.setStreamName("Blazepose")

    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("MaskedFrame")

    # define properties
    internal_fps = 30
    camRGB.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRGB.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRGB.setInterleaved(False)
    camRGB.setFps(internal_fps)

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setCamera("left")

    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setCamera("right")

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    # Align depth map to the perspective of RGB camera, on which inference is done
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(False)
    stereo.setLeftRightCheck(True)
    stereo.setConfidenceThreshold(230)
    stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    camRGB.preview.link(xoutRGB.input)
    
    # set blazepose module
    xyz = True
    internal_frame_height = 300
    no_pos_estimate = False
    blazepose_model = BlazeposeDepthaiModule(input_src="rgb", 
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
    img_w,img_h = blazepose_model.set_pipeline(pipeline,camRGB,stereo,xinFrame,xoutBlazepose)

    show_3d = False
    renderer = BlazeposeRenderer(
                blazepose_model, 
                show_3d=show_3d, 
                output=None)

    device = dai.Device()
    device.startPipeline(pipeline)
    qRgb = device.getOutputQueue("RGB")
    qBlazepose = device.getOutputQueue("Blazepose")
    inQ = device.getInputQueue("MaskedFrame")
    counter = 0
    fps = 0
    startTime = time.monotonic()
    while True:
        frame = qRgb.get().getCvFrame()
        img = dai.ImgFrame()
        img.setData(frame.transpose(2,0,1).flatten())
        img.setTimestamp(time.monotonic())
        img.setWidth(img_w)
        img.setHeight(img_h)
        img.setType(dai.RawImgFrame.Type.BGR888p)
        inQ.send(img)
        if qBlazepose.has():
            res = marshal.loads(qBlazepose.get().getData())
            body = blazepose_model.inference(res)
            frame = renderer.draw(frame, body)
            print(f'blazepose{counter}')
        
        else:
            continue
        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
        cv2.imshow("preview", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    device.close()