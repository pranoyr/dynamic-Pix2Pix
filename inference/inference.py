
import torchvision.transforms as transforms
import torch
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import cv2
from PIL import Image
import argparse
from models import GeneratorUNet

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, help="path to image")
parser.add_argument("--checkpoint", type=str, default="checkpoints/final_weights_exp1.pth", help="path to checkpoint")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = GeneratorUNet().to(device)
checkpoint = torch.load(args.checkpoint, map_location=device)
generator.load_state_dict(checkpoint['G_state_dict'])
generator.eval()

transform = transforms.Compose([
        transforms.Resize((256, 256)),
		transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

inverse_transform = transforms.Normalize(
					mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
					std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
					)



img = Image.open(args.img_path)
w, h = img.size

# Image A
img_A = img.crop((0, 0, w / 2, h)) # A

# Target B
img_B = img.crop((w / 2, 0, w, h)) 
img_B = np.array(img_B)/255
# covnert to float
img_B = img_B.astype(np.float32)
img_B = cv2.resize(img_B, (256, 256))

# Predicted Fake B
img_A = transform(img_A).unsqueeze(0).to(device)
fake_B = generator(img_A) 

###### visualise the output ######
img_A = inverse_transform(img_A[0]).permute(1,2,0).detach().cpu().numpy() 
fake_B = inverse_transform(fake_B[0]).permute(1,2,0).detach().cpu().numpy() 
# to BGR 
img_A = cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR)
fake_B = cv2.cvtColor(fake_B, cv2.COLOR_RGB2BGR)
img_B = cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR)
out = np.concatenate((img_A, fake_B, img_B), axis=1)
cv2.imshow("IMG A , Generated B , GT_B", out)
cv2.waitKey(0)
