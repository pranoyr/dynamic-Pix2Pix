
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
parser.add_argument("--img_path", type=str, default="images/14.jpg", help="path to image")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = GeneratorUNet().to(device)
checkpoint = torch.load("checkpoints/final_weights_exp1.pth")
generator.load_state_dict(checkpoint['G_state_dict'])
generator.eval()

transform = transforms.Compose([
		transforms.ToTensor(),
        transforms.Resize((256, 256)),
])



img = Image.open(args.img_path)
w, h = img.size

# Image A
img_A = img.crop((0, 0, w / 2, h)) # A

# Target B
img_B = img.crop((w / 2, 0, w, h)) 
img_B = np.array(img_B)/255
img_B = cv2.resize(img_B, (256, 256))

# Predicted Fake B
img_A = transform(img_A).unsqueeze(0).to(device)
fake_B = generator(img_A) 

###### visualise the output ######
img_A = img_A[0].permute(1,2,0).detach().cpu().numpy() 
fake_B = fake_B[0].permute(1,2,0).detach().cpu().numpy() 
out = np.concatenate((img_A, fake_B, img_B), axis=1)
cv2.imshow("IMG A , Generated B , GT_B", out)
cv2.waitKey(0)
