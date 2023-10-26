# hirarchical-human-action-recognition-based-human-robot-interaction
This repo will consist of three parts:
* Distance evaluation
* close pose estimation
* safe control limit

commitment log:
* run pipeline v0: can run pipeline, but no masked frame and blazepose might lose frames
* run pipeline v1: solve the problem of blazepose losing frames(actually it is still slow, but we won't see the lost frames any more)
* run pipeline v2: move the whole pipeline to the outside and try PSTMO(very slow)
* run pipeline v3: add mask
* try handpose and add human traj visualization: add handpose but we don't need such precise prediction, add human traj visualization(based on PSTMO visualization settings)

Human_traj link:
https://drive.google.com/drive/folders/1YUqPF1TJxWBESdfgY9P-f4x1rXedP34I?usp=drive_link

Noisome bug logs:
1. I kept meeting with this error:
```QObject::moveToThread: Current thread (0x562521cd61c0) is not the object's thread (0x5625213cf9c0).
Cannot move to target thread (0x562521cd61c0)

qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/peiqi/.local/lib/python3.9/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb, eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl.

Aborted (core dumped)
```

This has really bothered me for a lot of time! I have tried lots of solutions but none of them work. However, after I export DISPLAY=:0, this trouble solved! So I am just logging down here in case any similar bugs occur in the future.


