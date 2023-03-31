import torch
import utils

import random

import os
import numpy as np
import wandb
import argparse
from lib import GeneratorUNet, Discriminator, Trainer, get_config, utils


	

def main(cfg):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	generator = GeneratorUNet().to(device)
	discriminator = Discriminator().to(device)

	trainer = Trainer(cfg, generator, discriminator, device)
	trainer.fit()

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch 3D Face Reconstruction')
	parser.add_argument('--cfg', default='cfg/config.yaml', type=str, help='config file path')
	opt = parser.parse_args()

	cfg = get_config(opt.cfg)

	if cfg.TRAIN.DISTRIBUTED:
		if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
			rank = int(os.environ["RANK"])
			world_size = int(os.environ['WORLD_SIZE'])
			print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
		else:
			rank = -1
			world_size = -1
		torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
		torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
		torch.distributed.barrier()

	seed = cfg.SEED + utils.get_rank()
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	# cudnn.benchmark = True
	wandb.init(project=cfg.MODEL.NAME, config=cfg, name=cfg.EXP_NAME)
	main(cfg)


