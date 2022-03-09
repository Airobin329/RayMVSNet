import torch
from IPython.terminal.embed import embed
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from dataloader.mvs_dataset import MVSTrainSet
from networks.raymvsnet import RayMVSNet
from utils.utils import *
from collections import OrderedDict 
import argparse, os,  time, gc


cudnn.benchmark = True
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

is_distributed = num_gpus > 1



parser = argparse.ArgumentParser(description='Deep stereo using adaptive cost volume.')
parser.add_argument('--root_path', type=str, help='path to root directory.', default='./data/dtu')
parser.add_argument('--train_list', type=str, help='train scene list.', default='./dataloader/datalist/dtu/train.txt')
parser.add_argument('--save_path', type=str, help='path to save checkpoints.', default='./checkpoints')

parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.0016)
parser.add_argument('--lr_idx', type=str, default="10,12,14:0.5")

parser.add_argument('--loss_weights', type=str, default="0.5,1,2,4")
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--num_views', type=int, help='num of candidate views', default=2)
parser.add_argument('--num_patch', type=int, help='num of patchs', default=4)
parser.add_argument('--lamb', type=float, help='the interval coefficient.', default=1.5)
parser.add_argument('--net_configs', type=str, help='number of samples for each stage.', default='64,32,8')

parser.add_argument('--log_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency.')

parser.add_argument('--sync_bn', action='store_true',help='Sync BN.')
parser.add_argument('--opt_level', type=str, default="O0")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--local_rank", type=int, default=0)


args = parser.parse_args()

if args.sync_bn:
	import apex
	import apex.amp as amp

on_main = True

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main(args, model:nn.Module, optimizer, train_loader):
	milestones = list(map(lambda x: int(x) * len(train_loader), args.lr_idx.split(':')[0].split(',')))
	gamma = float(args.lr_idx.split(':')[1])
	scheduler = get_step_schedule_with_warmup(optimizer=optimizer, milestones=milestones, gamma=gamma)

	loss_weights = list(map(float, args.loss_weights.split(',')))

	for ep in range(args.epochs):
		model.train()
		for batch_idx, sample in enumerate(train_loader):
			for patch_idx in range(args.num_patch):
				tic = time.time()
				sample_cuda = dict2cuda(sample)
				optimizer.zero_grad()
				outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], sample_cuda["depth_labels"]['stage3'], patch_idx)
				loss = multi_stage_loss(outputs, sample_cuda["depth_labels"], sample_cuda["masks"], patch_idx, loss_weights)
				del outputs,sample_cuda
				torch.cuda.empty_cache()
				if is_distributed and args.sync_bn:
					with amp.scale_loss(loss, optimizer) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()

				optimizer.step()
				scheduler.step()

			log_index = (len(train_loader)) * ep + batch_idx
			if log_index % args.log_freq == 0:

				if on_main:
					print("Epoch {}/{}, Iter {}/{}, lr {:.6f}, train loss {:.2f},  time = {:.2f}".format(
						ep+1, args.epochs, batch_idx+1, len(train_loader),
						optimizer.param_groups[0]["lr"], loss,
						time.time() - tic))
			torch.cuda.empty_cache()
			gc.collect()

		if on_main and batch_idx % args.save_freq == 0:
			torch.save({"epoch": ep+1,
				"model": model.module.state_dict(),
				"optimizer": optimizer.state_dict()},
				"{}/model_{:06d}.ckpt".format(args.save_path, ep+1))


def distribute_model(args):
	def sync():
		if not dist.is_available():
			return
		if not dist.is_initialized():
			return
		if dist.get_world_size() == 1:
			return
		dist.barrier()

	if is_distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(
			backend="nccl", init_method="env://"
		)
		sync()


	model: torch.nn.Module = RayMVSNet(stage_configs=list(map(int, args.net_configs.split(","))),
	                                lamb=args.lamb)
	model.to(torch.device("cuda"))

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
	                       weight_decay=args.wd)

	train_set = MVSTrainSet(root_dir=args.root_path, data_list=args.train_list, num_views=args.num_views)


	if is_distributed:
		if args.sync_bn:
			model = apex.parallel.convert_syncbn_model(model)
			model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, )
			print('Convert BN to Sync_BN successful.')

		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank,)
		train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=dist.get_world_size(),
		                                                    rank=dist.get_rank())

	else:
		# model = nn.DataParallel(model, device_ids=[0,1,2])
		model = nn.DataParallel(model)
		train_sampler = None

	train_loader = DataLoader(train_set, args.batch_size, sampler=train_sampler, num_workers=1,
	                          drop_last=True, shuffle=not is_distributed)


	return model, optimizer, train_loader

def multi_stage_loss(outputs, labels, masks, patch_idx,weights):
	tot_loss = 0.
	for stage_id in range(2):
		depth_i = outputs["stage{}".format(stage_id+1)]["depth"]
		label_i = labels["stage{}".format(stage_id+1)]

		mask_i = masks["stage{}".format(stage_id+1)].bool()
		depth_loss = F.smooth_l1_loss(depth_i[mask_i], label_i[mask_i], reduction='mean')

		tot_loss += depth_loss * weights[stage_id]

	label_i = labels["stage3"]
	mask_i = masks["stage3"].bool()
    	
    	
	if patch_idx==0:
			label_i=label_i[:,0:256,0:320]
			mask_i=mask_i[:,0:256,0:320]
	if patch_idx==1:
			label_i=label_i[:,0:256,320:640]
			mask_i=mask_i[:,0:256,320:640]

	if patch_idx==2:
			label_i=label_i[:,256:512,0:320]
			mask_i=mask_i[:,256:512,0:320]

	if patch_idx==3:
			label_i=label_i[:,256:512,320:640]
			mask_i=mask_i[:,256:512,320:640]
	depth_f = outputs["stage_ray"]["final_depth"]

	depth_loss = F.smooth_l1_loss(depth_f[mask_i], label_i[mask_i], reduction='mean')


	tot_loss += depth_loss * weights[2]

	sdf_gt=outputs["stage_ray"]["sdf_gt"]
	sdf_feature = outputs["stage_ray"]["sdf_feature"]

	batchsize=sdf_gt.shape[0]
	sdf_feature=sdf_feature.view(batchsize,256,320,1,16).permute(0,3,4,1,2)
	mask_sdf=mask_i.unsqueeze(1).unsqueeze(1).repeat(1,1,16,1,1).view(batchsize,1,16,256,320)
	sdf_loss=F.l1_loss(sdf_feature[mask_sdf], sdf_gt[mask_sdf], reduction='mean')

	tot_loss += sdf_loss * weights[3]

	
	return tot_loss

if __name__ == '__main__':

	model, optimizer, train_loader = distribute_model(args)
	on_main = (not is_distributed) or (dist.get_rank() == 0)

	if on_main:
		mkdir_p(args.save_path)
		main(args=args, model=model, optimizer=optimizer, train_loader=train_loader)
