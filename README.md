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


