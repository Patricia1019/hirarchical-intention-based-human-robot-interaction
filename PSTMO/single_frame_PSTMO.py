import os
import time

from common.arguments import parse_args
from common.camera import *
from common.generators import *
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path
from common.inference_3d import *

from model.block.refine import refine
from model.stmo import Model
from tqdm import tqdm
import pdb
from common.visualization_frame import render_animation


from pathlib import Path
PATH_DIR = Path(__file__).resolve().parent

# from joints_detectors.openpose.main import generate_kpts as open_pose


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

add_path()


# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()

class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 14, 15, 16]

def get_detector_2d(detector_name):
    # def get_alpha_pose():
    #     from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
    #     return alpha_pose

    # def get_hr_pose():
    #     from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
    #     return hr_pose

    def get_mediapipe_pose():
        from joints_detectors.mediapipe.pose import generate_single_kpts as mediapipe_pose
        return mediapipe_pose

    detector_map = {
        # 'alpha_pose': get_alpha_pose,
        # 'hr_pose': get_hr_pose,
        # 'open_pose': open_pose
        'mediapipe_pose': get_mediapipe_pose,
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


class PSTMO:
    def __init__(self,detector_2d):
        self.args = parse_args()
        self.args.detector_2d = detector_2d
        self.args.evaluate = 'pretrained_h36m_detectron_coco.bin'

        self.model = {}
        self.model['trans'] = Model(self.args)
        model_dict = self.model['trans'].state_dict()
        no_refine_path = f"{PATH_DIR}/checkpoint/PSTMOS_no_refine_48_5137_in_the_wild.pth"
        pre_dict = torch.load(no_refine_path,map_location=torch.device('cpu'))
        for key, value in pre_dict.items():
            name = key[7:]
            model_dict[name] = pre_dict[key]
        self.model['trans'].load_state_dict(model_dict)
        print('PSTMO model loads done!')

        keypoints_symmetry = metadata['keypoints_symmetry']
        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

        receptive_field = self.args.frames
        self.pad = (receptive_field - 1) // 2  # Padding on each side
        self.causal_shift = 0

        self.detector_2d = get_detector_2d(self.args.detector_2d)

        self.rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)

    def inference_frame(self,frame):
        input_keypoints = []
        prediction = []
        # 2D kpts loads or generate
        frame_keypoints = np.array(self.detector_2d(frame))  ### detect 2d keypoints, around 40it/s, [frame,17,2]
        # normlization keypoints  Suppose using the camera parameter
        frame_keypoints = normalize_screen_coordinates(frame_keypoints[..., :2], w=1000, h=1002)
        frame_input_keypoints = frame_keypoints.copy()
        assert frame_input_keypoints.shape[0] == 1
        input_keypoints.append(frame_input_keypoints)

        # 3d keypoints detection
        gen = Evaluate_Generator(128, None, None, [frame_input_keypoints], self.args.stride,
                            pad=self.pad, causal_shift=self.causal_shift, augment=self.args.test_time_augmentation, shuffle=False,
                            kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left, joints_right=self.joints_right)
        frame_prediction = val(self.args, gen, self.model)
        prediction.append(frame_prediction)

        input_keypoints = np.array(input_keypoints).squeeze(1)
        prediction = np.array(prediction).squeeze(1)
        prediction = camera_to_world(prediction, R=self.rot, t=0)
        
        # save 3D joint points
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        return prediction,input_keypoints

    def render_prediction(self,prediction,input_keypoints):
        anim_output = {'Ours': prediction}
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)
        render_animation(input_keypoints, anim_output,
                     Skeleton(), 25, self.args.viz_bitrate, np.array(70., dtype=np.float32), self.args.viz_output,
                     limit=self.args.viz_limit, downsample=self.args.viz_downsample, size=self.args.viz_size,
                     input_video_path=None, viewport=(1000, 1002),
                     input_video_skip=self.args.viz_skip)

if __name__ == '__main__':
    PSTMO_model = PSTMO('mediapipe_pose')
    vid = cv2.VideoCapture('../input_vid/H013_GA_02_20210922_133913.mp4')
    video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(video_length)):
        ret, frame = vid.read()
        if not ret:
            break
        PSTMO_model.inference_frame(frame)