import numpy as np
import os, glob
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
	def __init__(self, root, transforms=None, mode="train"):
		self.transform = transforms
		self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

	def __getitem__(self, index):
		img = Image.open(self.files[index % len(self.files)])
		w, h = img.size
		img_A = img.crop((0, 0, w / 2, h))
		img_B = img.crop((w / 2, 0, w, h))
		
		img_A = self.transform(img_A)
		img_B = self.transform(img_B)
		return {"A": img_A, "B": img_B}

	def __len__(self):
		return len(self.files)
	

