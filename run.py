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
from BlazeposeDepthaiEdge_module import BlazeposeDepthaiModule


if __name__ == '__main__':
    # pseudo code

    # open camera
    # while True:
    #     frame = get_frame()
    #     humans_pos,confs = detect_human(frame)
    #     nearer_dist,nearer_human_pos = get_nearer_human(humans_pos)
    #     if nearer_dist < limit: # limit must take into v into consideration because human detection might recognize sth non-human as human
    #         masked_frame = mask_frame(nearer_human_pos)
    #         human_pose,human_depth = pose_estimation(masked_frame)

    # real code

    # path
    parentDir = Path(__file__).parent
    detection_nnPath = str((parentDir / Path('./models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())

    parser = argparse.ArgumentParser()
    parser.add_argument('--syncNN', action="store_true",
                    help="sync the output of detection network with the input of pose estimation network")
    parser.add_argument('--show', action="store_true",
                    help="show real time camera video")
    args = parser.parse_args()
    show = args.show
    syncNN = args.syncNN

    pipeline = dai.Pipeline()
    # define sources
    camRGB = pipeline.createColorCamera()
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    script = pipeline.createScript()
    pose_manip = pipeline.createImageManip()

    # define outputs
    xoutDetection = pipeline.create(dai.node.XLinkOut)
    xoutDetection.setStreamName("Detection")
    xoutRGB = pipeline.create(dai.node.XLinkOut)
    xoutRGB.setStreamName("RGB")
    xoutBlazepose = pipeline.create(dai.node.XLinkOut)
    xoutBlazepose.setStreamName("Blazepose")
    xoutMask = pipeline.create(dai.node.XLinkOut)
    xoutMask.setStreamName("Mask")
    xoutDebug = pipeline.create(dai.node.XLinkOut)
    xoutDebug.setStreamName("Debug")

    # define properties
    internal_fps = 30
    DET_INPUT_SIZE = (300,300)
    det_manip = pipeline.createImageManip()
    det_manip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
    det_manip.initialConfig.setKeepAspectRatio(False)
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

    spatialDetectionNetwork.setBlobPath(detection_nnPath)
    spatialDetectionNetwork.setConfidenceThreshold(0.5)
    spatialDetectionNetwork.input.setBlocking(False)
    spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
    spatialDetectionNetwork.setDepthLowerThreshold(100)
    spatialDetectionNetwork.setDepthUpperThreshold(5000)

    # Linking
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    camRGB.preview.link(det_manip.inputImage)
    det_manip.out.link(spatialDetectionNetwork.input)
        
    spatialDetectionNetwork.out.link(script.inputs['detections'])

    stereo.depth.link(spatialDetectionNetwork.inputDepth)

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
    img_w,img_h = blazepose_model.set_pipeline(pipeline,camRGB,stereo,script,xoutBlazepose)


    POSE_INPUT_SIZE = (img_w,img_h) # 512 288
    pose_manip.initialConfig.setResize(POSE_INPUT_SIZE[0], POSE_INPUT_SIZE[1])
    pose_manip.initialConfig.setKeepAspectRatio(True)
    if syncNN:
        spatialDetectionNetwork.passthrough.link(pose_manip.inputImage)
        pose_manip.out.link(script.inputs['frame'])
    else:
        camRGB.preview.link(script.inputs['frame'])

    show_3d = False
    renderer = BlazeposeRenderer(
                blazepose_model, 
                show_3d=show_3d, 
                output=None)

    # script.outputs['masked_frame'].link() # written in blazepose_model.set_pipeline()
    # script.outputs['detected_frame'].link(mask_manip.inputImage)
    # script.outputs['mask_img_config'].link(mask_manip.inputConfig)
    # script.outputs['detected_frame'].link(xoutMask.input)
    script.outputs['out_detections'].link(xoutDetection.input)
    script.outputs['rgb_frame'].link(xoutRGB.input)
    # script.outputs['frame_data'].link(xoutDebug.input) # debugging

    # set script # TODO
    '''
    input:
        frame: original rgb frame
        detections: human detection results
    output: 
        boxed_frame: when far away, output frame that has boxed where humans are
        masked_frame: when near, output masked_frame into Blazepose
    '''
    script.setScript("""
        while True:
            frame = node.io['frame'].get()
            detections_ = node.io['detections'].tryGet()
            send_flag = 0
            if detections_ is not None:
                detections = detections_.detections
                nearest_dist = 50000
                nearest_person = None
                for detection in detections:
                    if detection.label == 15 and detection.confidence > 0.6:
                        if detection.spatialCoordinates.z < 1000: # TODO
                            send_flag = 1
                            if detection.spatialCoordinates.z < nearest_dist:
                                nearest_dist = detection.spatialCoordinates.z
                                nearest_person = detection
                        elif send_flag != 1:
                            send_flag = 2
                     
            node.io['rgb_frame'].send(frame)
            if send_flag == 1: # pose estimation TODO
                # node.io['detected_frame'].send(frame)
                # mask_img_config = ImageManipConfig()
                # mask_img_config.setCropRect(nearest_person.xmin,nearest_person.ymin,\
                #      nearest_person.xmax,nearest_person.ymax)
                # node.io['mask_img_config'].send(mask_img_config)
                # buf = Buffer()
                # buf.setData()
                node.io['detected_frame'].send(frame)
            elif send_flag == 2: # detection
                node.io['out_detections'].send(detections_)
    """)

    device = dai.Device()
    device.startPipeline(pipeline) 
    calib_data = device.readCalibration()
    calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.CAM_A)
    print(f"RGB calibration lens position: {calib_lens_pos}")
    camRGB.initialControl.setManualFocus(calib_lens_pos)
    if 1:
        qDetection = device.getOutputQueue("Detection")
        qBlazepose = device.getOutputQueue("Blazepose")
        qRgb = device.getOutputQueue("RGB")
        qMask = device.getOutputQueue("Mask")
        qDebug = device.getOutputQueue("Debug")
        
        counter = 0
        fps = 0
        startTime = time.monotonic()
        color = (255, 255, 255)
        while True:
            frame = qRgb.get().getCvFrame()
            if qDebug.has():
                debug_info = qDebug.get()
                print(debug_info)
            if qDetection.has():
                detections = qDetection.get()
                detections = detections.detections
                print(f'detection{counter}')

                if show:
                    height = frame.shape[0] 
                    width  = frame.shape[1] 
                    for detection in detections:
                        if detection.label == 15 and detection.confidence > 0.6:
                            x1 = int(detection.xmin * width)
                            x2 = int(detection.xmax * width)
                            y1 = int(detection.ymin * height)
                            y2 = int(detection.ymax * height)
                            label = 'person'
                            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), cv2.FONT_HERSHEY_SIMPLEX)

            elif qBlazepose.has():
                # frame = qMask.get().getCvFrame()
                res = marshal.loads(qBlazepose.get().getData())
                print(f'blazepose{counter}')
                body = blazepose_model.inference(res)
                if show:
                    frame = renderer.draw(frame, body)
            
            else:
                continue

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            if show:
                cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))
                cv2.imshow("preview", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

    device.close()
                                                                                                                                            
                                                                                                                                            



