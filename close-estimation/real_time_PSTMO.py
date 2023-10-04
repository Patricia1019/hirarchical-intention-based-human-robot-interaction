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

def main(args):
    detector_2d = get_detector_2d(args.detector_2d)
    assert detector_2d, 'detector_2d should be in ({alpha, hr, open, media}_pose)'

    video_file = args.viz_video
    vid = cv2.VideoCapture(video_file)
    video_length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    
    model = {}
    model['trans'] = Model(args)
    ckpt, time1 = ckpt_time(time0)
    print('-------------- load data spends {:.2f} seconds'.format(ckpt))

    model_dict = model['trans'].state_dict()
    no_refine_path = "checkpoint/PSTMOS_no_refine_48_5137_in_the_wild.pth"
    pre_dict = torch.load(no_refine_path,map_location=torch.device('cpu'))
    for key, value in pre_dict.items():
        name = key[7:]
        model_dict[name] = pre_dict[key]
    model['trans'].load_state_dict(model_dict)
    ckpt, time2 = ckpt_time(time1)
    print('-------------- load 3D model spends {:.2f} seconds'.format(ckpt))

    receptive_field = args.frames
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    print('Extracting...')
    input_keypoints = []
    prediction = []
    for i in tqdm(range(video_length)):
        ret, frame = vid.read()
        if not ret:
            break
        # 2D kpts loads or generate
        frame_keypoints = np.array(detector_2d(frame))  ### detect 2d keypoints, around 40it/s, [frame,17,2]
         # normlization keypoints  Suppose using the camera parameter
        frame_keypoints = normalize_screen_coordinates(frame_keypoints[..., :2], w=1000, h=1002)
        frame_input_keypoints = frame_keypoints.copy()
        assert frame_input_keypoints.shape[0] == 1
        input_keypoints.append(frame_input_keypoints)

        gen = Evaluate_Generator(128, None, None, [frame_input_keypoints], args.stride,
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation, shuffle=False,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        frame_prediction = val(args, gen, model)
        prediction.append(frame_prediction)

    input_keypoints = np.array(input_keypoints).squeeze(1)
    prediction = np.array(prediction).squeeze(1)
    prediction = camera_to_world(prediction, R=rot, t=0)
    
    # save 3D joint points
    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])
    np.save(f'outputs/test_3d_output_{args.video_name}_postprocess.npy', prediction, allow_pickle=True)
    
    anim_output = {'Ours': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

    # pdb.set_trace()
    print('Rendering...')
    from common.visualization import render_animation
    render_animation(input_keypoints, anim_output,
                     Skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
                     limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                     input_video_path=args.viz_video, viewport=(1000, 1002),
                     input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    print('total spend {:2f} second'.format(ckpt))


def inference_video(video_path, detector_2d):
    """
    Do image -> 2d points -> 3d points to video.
    :param detector_2d: used 2d joints detector. Can be {alpha_pose, hr_pose}
    :param video_path: relative to outputs
    :return: None
    """
    args = parse_args()

    args.detector_2d = detector_2d
    dir_name = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    args.video_name = basename[:basename.rfind('.')]
    args.viz_video = video_path
    # args.viz_export = f'{dir_name}/{args.detector_2d}_{video_name}_data.npy'
    args.viz_output = f'./outputs/{args.detector_2d}_{args.video_name}_realtime.mp4'
    # args.viz_limit = 20
    #args.input_npz = 'outputs/alpha_pose_test/test.npz'

    args.evaluate = 'pretrained_h36m_detectron_coco.bin'

    with Timer(video_path):
        main(args)

if __name__ == '__main__':
    inference_video('../input_vid/H013_GA_02_20210922_133913.mp4', 'mediapipe_pose')