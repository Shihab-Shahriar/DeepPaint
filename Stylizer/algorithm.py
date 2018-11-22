import os
import time 
from tqdm import tqdm_notebook as tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import random  
import numpy as np

import torch 
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image

from Stylizer.AbsModels import Encoder,Decoder
from Stylizer.AdaIN import stylizeAdaIN


MODELS = ['cezanne','el-greco','monet','picasso','van-gogh']


enc = Encoder().cuda().eval()
dec = Decoder().cuda().eval()
state_dicts = {}
for mod in MODELS:
    state_dicts[f"{mod}e"] = torch.load(f"weights/model_{mod}/enc.pth")
    state_dicts[f"{mod}d"] = torch.load(f"weights/model_{mod}/dec.pth")

print("Enc/dec instantiated....")

def stylize(Ic,Is,info):
	print("INFO:",info)
	if 'model_name' not in info:
		return stylizeAdaIN(Ic,Is,info['slider']/10)
        
	enc.load_state_dict(state_dicts[f"{info['model_name']}e"])
	dec.load_state_dict(state_dicts[f"{info['model_name']}d"])
	img = transforms.ToTensor()(Ic)
	arr = img.unsqueeze(0).cuda()
	arr = arr*2-1

	with torch.no_grad():
	    f = enc(arr)
	    arr = dec(f)

	arr = (arr+1)/2
	#arr -= arr.min()
	#arr /= arr.max()
	arr = arr.squeeze(0).cpu().numpy().transpose(1,2,0)*255
	return Image.fromarray(arr.clip(0,255).astype("uint8"))
