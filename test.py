
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from IPython.terminal.embed import embed

from dataloader.mvs_dataset import MVSTestSet
from networks.raymvsnet import RayMVSNet
from utils.utils import dict2cuda, dict2numpy, mkdir_p, save_cameras, write_pfm
from fusion import filter_depth
import numpy as np
import argparse, os, gc
from PIL import Image
import os.path as osp
from collections import *


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Test UCSNet.')

parser.add_argument('--root_path', type=str, help='path to root directory.', default='./data/dtu/Eval')
parser.add_argument('--test_list', type=str, help='testing scene list.', default='./dataloader/datalist/dtu/test.txt')
parser.add_argument('--save_path', type=str, help='path to save depth maps.', default='./outputs')

#test parameters
parser.add_argument('--max_h', type=int, help='image height', default=1080)
parser.add_argument('--max_w', type=int, help='image width', default=1920)
parser.add_argument('--num_patch', type=int, help='num of patchs', default=4)
parser.add_argument('--num_views', type=int, help='num of candidate views', default=3)
parser.add_argument('--lamb', type=float, help='the interval coefficient.', default=1.5)
parser.add_argument('--net_configs', type=str, help='number of samples for each stage.', default='64,32,8')
parser.add_argument('--ckpt', type=str, help='the path for pre-trained model.', default='./model.ckpt')


args = parser.parse_args()


def main(args):

    testset = MVSTestSet(root_dir=args.root_path, data_list=args.test_list,
                         max_h=args.max_h, max_w=args.max_w, num_views=args.num_views)
    test_loader = DataLoader(testset, 1, shuffle=False, num_workers=1, drop_last=False)

    # build model
    model = RayMVSNet(stage_configs=list(map(int, args.net_configs.split(","))),
                   lamb=args.lamb)

    print("Loading model {} ...".format(args.ckpt))


    state_dict = torch.load(args.ckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    print('Success!')

    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    for batch_idx, sample in enumerate(test_loader):
        scene_name = sample["scene_name"][0]
        frame_idx = sample["frame_idx"][0][0]
        scene_path = osp.join(args.save_path, scene_name)
        h=sample["depth"].shape[1]
        w=sample["depth"].shape[2]
        
        depth_sdf=np.zeros((1,h,w),dtype="float32")
        for patch_idx in range(args.num_patch):


            sample_cuda = dict2cuda(sample)

            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], sample_cuda["depth"], patch_idx)       
            outputs = dict2numpy(outputs)
            del sample_cuda


            print('Finished {}_{}/{}, time: {:.2f}s'.format(batch_idx+1, patch_idx, len(test_loader)))
            if patch_idx==0:
                depth_sdf[:,0:h//2,0:w//2]=outputs['stage_ray']["final_depth"]
            if patch_idx==1:
                depth_sdf[:,0:h//2,w//2:w]=outputs['stage_ray']["final_depth"]
            if patch_idx==2:
                depth_sdf[:,h//2:h,0:w//2]=outputs['stage_ray']["final_depth"]
            if patch_idx==3:
                depth_sdf[:,h//2:h,w//2:w]=outputs['stage_ray']["final_depth"]            


        rgb_path = osp.join(scene_path, 'rgb')
        mkdir_p(rgb_path)
        depth_sdf_path = osp.join(scene_path, 'depth_sdf')
        mkdir_p(depth_sdf_path)
        cam_path = osp.join(scene_path, 'cam')
        mkdir_p(cam_path)
        conf_path = osp.join(scene_path, 'confidence')
        mkdir_p(conf_path)


        ref_img = sample["imgs"][0, 0].numpy().transpose(1, 2, 0) * 255
        ref_img = np.clip(ref_img, 0, 255).astype(np.uint8)
        Image.fromarray(ref_img).save(rgb_path+'/{:08d}.png'.format(frame_idx))

        cam = sample["proj_matrices"]["stage3"][0, 0].numpy()
        save_cameras(cam, cam_path+'/cam_{:08d}.txt'.format(frame_idx))
        write_pfm(conf_path+'/conf_{:08d}.pfm'.format(frame_idx), outputs["stage1"]["confidence"][0])
        write_pfm(depth_sdf_path+"/dep_{:08d}.pfm".format(frame_idx), depth_sdf[0])
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':

	with torch.no_grad():
	    main(args)

