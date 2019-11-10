
#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors, write_obj_with_colors_texture, \
    create_unwraps, process_uv, scale_tcoords
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose #,angle2matrix_3ddfa
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
from utils.ddfa import _parse_param
from utils.params import *
import argparse
import torch.backends.cudnn as cudnn
from mesh.render import render_colors
STD_SIZE = 120
import infer_uv_gan 

def main(args):
    # 1. load pre-trained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'
    
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()
    
    # 2. load pre-trained model uv-gan
    if args.uvgan:
        if args.checkpoint_uv_gan == "":
            print("Specify the path to checkpoint uv_gan")
            exit()
        uvgan = infer_uv_gan.UV_GAN(args.checkpoint_uv_gan)

    # 3. load dlib model for face detection and landmark used for face cropping
    if args.dlib_landmark:
        dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
        face_regressor = dlib.shape_predictor(dlib_landmark_model)
    if args.dlib_bbox:
        face_detector = dlib.get_frontal_face_detector()

    # 4. forward
    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    for img_fp in args.files:
        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

        pts_res = []
        Ps = []  # Camera matrix collection
        poses = []  # pose collection, [todo: validate it]
        vertices_lst = []  # store multiple face vertices
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
            if args.dlib_landmark:
                # - use landmark for cropping
                pts = face_regressor(img_ori, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                roi_box = parse_roi_box_from_landmark(pts)
            else:
                # - use detected face bbox
                bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                roi_box = parse_roi_box_from_bbox(bbox)

            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            # two-step for more accurate bbox to crop face
            if args.bbox_init == 'two':
                roi_box = parse_roi_box_from_landmark(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)
            P, pose = parse_pose(param)

            Ps.append(P)
            poses.append(pose)

            if args.dump_obj:
                vertices = predict_dense(param, roi_box) 
                vertices_lst.append(vertices)
                wfp = '{}_{}.obj'.format(img_fp.replace(suffix, ''), ind)
                colors = get_colors(img_ori, vertices)

                p, offset, alpha_shp, alpha_exp = _parse_param(param)

                vertices = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
                vertices = vertices.T
                tri = tri.T - 1
                print('Dump obj with sampled texture to {}'.format(wfp))
                unwraps = create_unwraps(vertices)
                h, w = args.height, args.width
                tcoords = process_uv(unwraps[:,:2], h, w)
                texture = render_colors(tcoords, tri, colors, h, w, c=3).astype('uint8')
                scaled_tcoords = scale_tcoords(tcoords)
                if args.uvgan:
                   texture =  uvgan.infer(texture)
                else:
                   texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                vertices, colors, uv_coords = vertices.astype(np.float32).copy(), colors.astype(np.float32).copy(), scaled_tcoords.astype(np.float32).copy()
                write_obj_with_colors_texture(wfp, vertices, colors, tri, texture*255.0, uv_coords)

            ind += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', default=[''], nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-c', '--checkpoint_uv_gan', default = '',type=str, help='checkpoint UV GAN')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')
    parser.add_argument('--height', default=256, type=int, help='height of image')
    parser.add_argument('--width', default=256, type=int, help='width of image')
    parser.add_argument('-u', '--uvgan', default='false', type=str2bool, help='using to uv_gan to rotate profile face to frontal face')

    args = parser.parse_args()
    print(args)
    main(args)
