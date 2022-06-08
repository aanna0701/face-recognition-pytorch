import sys
from pathlib import Path
# p = Path.cwd().parent
# sys.path.append(str(p))
p = '/workspace'
sys.path.append(str(p))
from facedetection.src.Models.Network import FaceDetector
from facedetection.src.utils.box_utils import ProcOutput, bbox2square
from facedetection.src.layers.layers import make_prior_box
import argparse
import cv2
import numpy as np
import torch


# =========================================== Arguments ===========================================

def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='Make labels(csv format) for FAS')
    parser.add_argument('--input_dir', default=r'/workspace/dataset/FR/train/lfw/imgs')
    parser.add_argument('--weight', default=r'./ssh_mobilenetv1_relu_latest.pth')
    parser.add_argument('--box_size', default=200)
    parser.add_argument('--landmark', action='store_true')
    args = parser.parse_args()

    return args

# ==================================================================================================

if __name__ == "__main__":
    
    ########### Args ########### 
    global args
    args = parse_args()
    ########### Parameter Settig ###########
    ## Input images
    # Path class ($input_dir/)
    p = Path(args.input_dir)
    save_dir = p.parent / 'imgs_fd'
    save_dir.mkdir(parents=True, exist_ok=True)
    inputs = [(img.name, img.parent.name, img) for img in p.glob('**/*.jpg')]
    
    for file_name, save_folder, input_path in inputs:
        save_path = save_dir / save_folder / file_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(input_path)

        net = FaceDetector('ssh_mobilenetv1', num_landmark=5, pretrain=False)
        net.to('cuda')
        net.eval()

        weights = torch.load(args.weight, map_location='cuda')
        net.load_state_dict(weights, strict=True)
        
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        steps = [8, 16, 32]
        variance = [0.1, 0.2]
        confidence_threshold = 0.8
        nms_threshold = 0.5
        keep_top_k = 10
        
    
        img = cv2.imread(str(input_path))

        bboxs, scores, landms = net.predict(img)
        H, W, _ = np.shape(img)
        
        # ========================
        # make prior box
        # ========================
        priors = make_prior_box([H, W], min_sizes, steps, clip=False)
        priors = priors.to('cuda')
        
        # ========================
        # Post process output
        # ========================
        proc_output = ProcOutput(variance, confidence_threshold, nms_threshold,
                                priors, [H, W], 'cuda', resize=1)
        my_output = proc_output({'cls': scores, 'bbox': bboxs, 'ldm': landms})

        bboxs = my_output[1][:keep_top_k, 0:4].cpu().data.numpy() * np.tile((1, 1), 2)    
        
        # ========================
        # Crop
        # ========================   
        if len(bboxs) == 0:
            continue
        
        
        b_size = ((bboxs[0][2] - bboxs[0][0]), (bboxs[0][3] - bboxs[0][1]))       
        
        # landmarks
        landms_np = my_output[-1][0].cpu().data.numpy()
        land_2d = np.reshape(landms_np, (5, 2))
        land_2d = [(int(x[0]), int(x[1])) for x in land_2d]
        land_map = np.zeros((H, W))
        for i, landmar in enumerate(land_2d):
            cv2.rectangle(land_map, (landmar[0]-int(b_size[0]/16), landmar[1]-int(b_size[1]/16)), (landmar[0]+int(b_size[0]/16), landmar[1]+int(b_size[1]/16)), 255, -1)
            
        
        # make box square from rectangle
        sboxs = bbox2square(img, bboxs[0])
        
        # crop
        img_cropped = img[sboxs[1]:sboxs[3], sboxs[0]:sboxs[2]]
        
        # save outputs
        cv2.imwrite(str(save_path), img_cropped)
    
    
    print('DONE!!!!')