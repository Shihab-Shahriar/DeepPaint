import numpy as np
from skimage import color
from PIL import Image
from typing import List
import torch
from Colorizer.model import SIGGRAPHGenerator

def lab2rgb_transpose(img_l, img_ab):
    #print("labtoRGB:",img_l.max(),img_l.min(),img_ab.max(),img_ab.min())
    pred_lab = np.concatenate((img_l, img_ab), axis=0).transpose((1, 2, 0))
    pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
    return pred_rgb

def run(img_rgb,inp_ab,inp_mask):
    lab_image = color.rgb2lab(img_rgb).transpose((2,0,1))
    #print("LABBB:",lab_image.shape,img_rgb.size)
    img_l = lab_image[[0],...]
    img_l -= 50.0

    with torch.no_grad():
        out_ab = model(img_l,inp_ab,inp_mask)
    img = lab2rgb_transpose(img_l+50.0,out_ab.detach().cpu().numpy()[0])

    return Image.fromarray(img)

def put_point(input_ab,mask,loc,p,val):
    val = val[:,np.newaxis,np.newaxis]
    input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = val
    mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1
    return (input_ab,mask)

def convRGB(rgb):
    im = [[rgb]]
    im = np.array(im)/255
    return color.rgb2lab(im)[0][0]

def colorize(img:Image.Image,points:List):
    #print("POINTS SIZE:",len(points),img.size)
    if img.size[-1]!=3:
        img = img.convert("RGB")
    inp_ab = np.zeros((2,)+img.size)
    inp_mask = np.zeros((1,)+img.size)
    for pos,col,r in points:
        col = convRGB(col)[1:]
        put_point(inp_ab,inp_mask,pos,r,col)
    img = run(img,inp_ab,inp_mask)
    return img

model = SIGGRAPHGenerator().cuda().eval()
state = torch.load("d:/DeepPaint/server/weights/colorizer.pth")
if hasattr(state,'_metadata'):
    del state._metadata
model.load_state_dict(state)


