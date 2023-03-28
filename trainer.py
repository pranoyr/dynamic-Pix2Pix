import torch
import numpy as np
import wandb
import os
import sys

import torchvision.transforms as transforms
from dataset import ImageDataset
from torch.nn.functional import mse_loss as criterion_GAN
from torch.nn.functional import l1_loss as criterion_pixelwise
import cv2
from models import weights_init_normal


class Trainer():
	def __init__(self, cfg, generator, discriminator, device):
		self.cfg = cfg
		self.device = device


		self.G_without_ddp = generator
		self.D_without_ddp = discriminator

		if self.cfg.TRAIN.DISTRIBUTED:
			self.G = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False)
			self.D = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False)
		else:
			self.G = generator
			self.D = discriminator

		self.patch = (1, self.cfg.DATA.IMG_SIZE // 2 ** 4, self.cfg.DATA.IMG_SIZE // 2 ** 4)
		
		self.start_epoch = 0
		

		self.prepare_data()
		self.get_training_config()
		self.init_model()


	def prepare_data(self):
		transform = transforms.Compose([
					transforms.Resize((self.cfg.DATA.IMG_SIZE, self.cfg.DATA.IMG_SIZE)),
				   	transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				])
		
		self.inverse_transform = transforms.Normalize(
					mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
					std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
					)


		train_dataset = ImageDataset(self.cfg.DATA.ROOT_DIR, transforms=transform, mode='train')
		val_dataset = ImageDataset(self.cfg.DATA.ROOT_DIR, transforms=transform, mode='val')
		self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=self.cfg.DATA.NUM_WORKERS)
		self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.cfg.DATA.BATCH_SIZE, shuffle=True, num_workers=self.cfg.DATA.NUM_WORKERS)
		print("train dataset size: ", len(train_dataset))
		
	def get_training_config(self):
		if self.cfg.TRAIN.OPTIMIZER.NAME == 'adam':
			self.optimizer_G = torch.optim.Adam(self.G_without_ddp.parameters(), lr=self.cfg.TRAIN.BASE_LR, betas=(0.5, 0.999))
			self.optimizer_D = torch.optim.Adam(self.D_without_ddp.parameters(), lr=self.cfg.TRAIN.BASE_LR, betas=(0.5, 0.999))
		
	def init_model(self):
		self.G_without_ddp.apply(weights_init_normal)
		self.D_without_ddp.apply(weights_init_normal)


	def adjust_optim(self, step):
		decay_epochs = self.cfg.TRAIN.LR_SCHEDULER.DECAY_EPOCHS
		if step >= decay_epochs:
			# decay the lr linearly to zero
			lr = self.cfg.TRAIN.BASE_LR * (1 - (step - decay_epochs) / (self.cfg.TRAIN.EPOCHS - decay_epochs))
			self.optimizer_G.param_groups[0]['lr'] = lr
			self.optimizer_D.param_groups[0]['lr'] = lr

	
	def save_checkpoint(self, epoch, filename, is_best=False):
		if is_best:
			checkpoint = {
				'epoch': epoch,
				'G_state_dict': self.G_without_ddp.state_dict(),
				'D_state_dict': self.D_without_ddp.state_dict(),
			}
			torch.save(checkpoint, filename)

	def sample_images(self):
		"""Saves a generated sample from the validation set"""
		imgs = next(iter(self.val_loader))
		real_A = imgs["A"].to(self.device)

		with torch.no_grad():
			fake_B = self.G(real_A)

		realA = self.inverse_transform(real_A[0]).permute(1,2,0).detach().cpu().numpy() 
		fake_B = self.inverse_transform(fake_B[0]).permute(1,2,0).detach().cpu().numpy() 
		out = np.concatenate((realA, fake_B), axis=1)
		output_path = self.cfg.OUTPUT_DIR + f'/result_{self.cfg.EXP_NAME}.png'
		cv2.imwrite(output_path, out*255)

	
	def data_cyc(self, batch):
		real_A = batch["A"].to(self.device, non_blocking=True)
		real_B = batch["B"].to(self.device, non_blocking=True)

		# Adversarial ground truths
		valid = torch.ones((real_A.size(0), *self.patch)).to(self.device, non_blocking=True)
		fake = torch.zeros((real_A.size(0), *self.patch)).to(self.device, non_blocking=True)

		# ------------------
		#  Train Generators
		# ------------------

		self.optimizer_G.zero_grad()

		# GAN loss
		fake_B = self.G(real_A)
		pred_fake = self.D(fake_B, real_A)
		loss_GAN = criterion_GAN(pred_fake, valid)
		# Pixel-wise loss
		loss_pixel = criterion_pixelwise(fake_B, real_B)

		# Total loss
		loss_G = loss_GAN + self.cfg.MODEL.PIXEL_LOSS_WEIGHT * loss_pixel

		loss_G.backward()

		self.optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		self.optimizer_D.zero_grad()

		# Real loss
		pred_real = self.D(real_B, real_A)
		loss_real = criterion_GAN(pred_real, valid)

		# Fake loss
		pred_fake = self.D(fake_B.detach(), real_A)
		loss_fake = criterion_GAN(pred_fake, fake)

		# Total loss
		loss_D = 0.5 * (loss_real + loss_fake)

		loss_D.backward()
		self.optimizer_D.step()

		metrics = {
			"loss_D": loss_D.item(),
			"loss_G": loss_G.item(),
			"loss_pixel": loss_pixel.item(),
			"loss_GAN": loss_GAN.item(),
		}

		return metrics
	

	def noise_cycle(self, batch):
		# sample a (9,256,256) matrix from uniform distribution
		noise_input = np.random.uniform(-1, 1, (batch["A"].size(0), 3, self.cfg.DATA.IMG_SIZE, self.cfg.DATA.IMG_SIZE))
		noise_input = torch.from_numpy(noise_input).float()

		real_A = noise_input.to(self.device, non_blocking=True)
		real_B = batch["B"].to(self.device, non_blocking=True)

		# Adversarial ground truths
		valid = torch.ones((real_A.size(0), *self.patch)).to(self.device, non_blocking=True)
		fake = torch.zeros((real_A.size(0), *self.patch)).to(self.device, non_blocking=True)

		# ------------------
		#  Train Generators
		# ------------------

		self.optimizer_G.zero_grad()

		# GAN loss
		fake_B = self.G(real_A, freeze_encoder=True)
		pred_fake = self.D(fake_B, real_A)
		loss_GAN = criterion_GAN(pred_fake, valid)

		# Total loss
		loss_G = loss_GAN * self.cfg.MODEL.NOISE_LOSS_WEIGHT

		loss_G.backward()

		self.optimizer_G.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		self.optimizer_D.zero_grad()

		# Real loss
		pred_real = self.D(real_B, real_A)
		loss_real = criterion_GAN(pred_real, valid)

		# Fake loss
		pred_fake = self.D(fake_B.detach(), real_A)
		loss_fake = criterion_GAN(pred_fake, fake)

		# Total loss
		loss_D = 0.5 * (loss_real + loss_fake)
		loss_D = loss_D * self.cfg.MODEL.NOISE_LOSS_WEIGHT

		loss_D.backward()
		self.optimizer_D.step()
			

	def fit(self):
		self.G.train()
		self.D.train()

		for epoch in range(self.start_epoch, self.cfg.TRAIN.EPOCHS):
			for idx, batch in enumerate(self.train_loader):
				metrics = self.data_cyc(batch)
				self.noise_cycle(batch)
		
				sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]"
									% (
										epoch,
										self.cfg.TRAIN.EPOCHS,
										idx,
										len(self.train_loader),
										metrics["loss_D"],
										metrics["loss_G"],
										metrics["loss_pixel"],
										metrics["loss_GAN"],
										
									)
								)

			if epoch % self.cfg.LOG_FREQ == 0:
				self.G.eval()
				self.sample_images()
				self.G.train()

				lr = self.optimizer_G.param_groups[0]['lr']
				wandb.log({'loss_D': metrics["loss_D"] , 'loss_G': metrics["loss_G"], 'loss_pixel': metrics["loss_pixel"], 'loss_GAN': metrics["loss_GAN"]}, step = epoch)
				wandb.log({'lr': lr}, step = epoch)

		
			if epoch % self.cfg.SAVE_FREQ == 0:
				# save model
				# checkpoint_iter10_trial1.pth
				checkpoint_name = self.cfg.CKPT_DIR + f"/checkpoint_epoch{epoch}_{self.cfg.EXP_NAME}.pth"
				self.save_checkpoint(epoch, checkpoint_name,  is_best=True)
				print("model saved")
		

			self.adjust_optim(epoch)

		# save final model
		checkpoint_name = self.cfg.CKPT_DIR + f"/final_weights_{self.cfg.EXP_NAME}.pth"
		self.save_checkpoint(epoch, checkpoint_name, is_best=True)
		print("model saved")
