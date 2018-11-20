import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from sklearn.cluster import KMeans
from scipy.ndimage.interpolation import zoom
from PIL import Image
from typing import List

import os
import torch
import torch.nn as nn

def lab2rgb_transpose(img_l, img_ab):
    ''' INPUTS
            img_l     1xXxX     [0,100]
            img_ab     2xXxX     [-100,100]
        OUTPUTS
            returned value is XxXx3 '''
    print("labtoRGB:",img_l.max(),img_l.min(),img_ab.max(),img_ab.min())
    pred_lab = np.concatenate((img_l, img_ab), axis=0).transpose((1, 2, 0))
    pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
    return pred_rgb


def rgb2lab_transpose(img_rgb):
    ''' INPUTS
            img_rgb XxXx3
        OUTPUTS
            returned value is 3xXxX '''
    return color.rgb2lab(img_rgb).transpose((2, 0, 1))




class SIGGRAPHGenerator(nn.Module):
    def __init__(self, dist=False):
        super(SIGGRAPHGenerator, self).__init__()
        self.dist = dist
        use_bias = True
        norm_layer = nn.BatchNorm2d

        # Conv1
        model1 = [nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]
        # add a subsampling operation

        # Conv2
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]
        # add a subsampling layer operation

        # Conv3
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]
        # add a subsampling layer operation

        # Conv4
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        # Conv5
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        # Conv6
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        # Conv7
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        # Conv7
        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]
        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        model8 = [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [norm_layer(256), ]

        # Conv9
        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # add the two feature maps above

        model9 = [nn.ReLU(True), ]
        model9 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model9 += [nn.ReLU(True), ]
        model9 += [norm_layer(128), ]

        # Conv10
        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # add the two feature maps above

        model10 = [nn.ReLU(True), ]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias), ]
        model10 += [nn.LeakyReLU(negative_slope=.2), ]

        # classification output
        model_class = [nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ]

        # regression output
        model_out = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ]
        model_out += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'), ])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1), ])

    def forward(self, input_A, input_B, mask_B):
        # input_A \in [-50,+50]
        # input_B \in [-110, +110]
        # mask_B \in [0, +0.5]

        print("A:", input_A.shape)
        print("B:", input_B. shape)
        print("mask:", mask_B.shape)


        #input_A = np.load("e:/A.npy")
        #input_B = np.load("e:/B.npy")
        #mask_B = np.load("e:/mask.npy")

        print("A:", input_A.mean(), input_A.std(), input_A.min(), input_A.max())
        try:
            print("B:", (input_B[input_B!=0]).mean(), (input_B[input_B!=0]).std(), (input_B[input_B!=0]).min(),
              (input_B[input_B != 0]).max())
            print("Mask:", (mask_B[mask_B!=0]).mean(), (mask_B[mask_B!=0]).std(),(mask_B[mask_B!=0]).min(),
              (mask_B[mask_B != 0]).max())
        except:
            print("Zero ")

        input_A = torch.Tensor(input_A).cuda()[None, :, :, :]
        input_B = torch.Tensor(input_B).cuda()[None, :, :, :]
        mask_B = torch.Tensor(mask_B).cuda()[None, :, :, :]

        conv1_2 = self.model1(torch.cat((input_A / 100., input_B / 110., mask_B - .5), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        print("AT MIDDLE:",conv8_3.mean(),conv8_3.std(),conv8_3.min(),conv8_3.max())

        if(self.dist):
            out_cl = self.upsample4(self.softmax(self.model_class(conv8_3) * .2))

            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

            return (out_reg * 110, out_cl)
        else:
            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
            return out_reg * 110

model = SIGGRAPHGenerator(True).cuda().eval()
state = torch.load("d:/DeepPaint/weights/colorizer.pth")
if hasattr(state,'_metadata'):
    del state._metadata
model.load_state_dict(state)

with open("d:/logs/log_deep.txt","w") as log:
    for n,par in model.named_parameters():
        s = f"{n},{par.mean()},{par.std()},{par.min()},{par.max()}"
        log.write(s+"\n")

col_dist = None

def findABCentroids(dist_ab,h,w,K=9):
    cmf = np.cumsum(dist_ab[:, h, w])  # CMF
    cmf = cmf / cmf[-1]
    cmf_bins = cmf

    rnd_pts = np.random.uniform(low=0, high=1.0, size=25000)
    inds = np.digitize(rnd_pts, bins=cmf_bins)

    pts_in_hull = np.array(np.meshgrid(np.arange(-110, 120, 10), np.arange(-110, 120, 10))).reshape((2, 529)).T
    rnd_pts_ab = pts_in_hull[inds, :]
    kmeans = KMeans(n_clusters=K).fit(rnd_pts_ab)
    return kmeans.cluster_centers_

def run(img_rgb,inp_ab,inp_mask):
    lab_image = color.rgb2lab(img_rgb).transpose((2, 0, 1))
    img_l = lab_image[[0],...]
    img_ab = lab_image[1:,...]
    img_l -= 50.0

    with torch.no_grad():
        out_ab,col_dist = model(img_l,inp_ab,inp_mask)
    img = lab2rgb_transpose(img_l+50.0,out_ab.detach().cpu().numpy()[0])
    print("OUT-AB:",out_ab.mean(),out_ab.std(),out_ab.min(),out_ab.max())
    print("OUT IMG-L:",img.mean(),img.std(),img.min(),img.max())

    return Image.fromarray(img),col_dist

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
    global col_dist
    print("POINTS SIZE:",len(points))
    arr = np.array(img)
    ab = np.abs(arr)
    print("In:",arr.mean(),arr.min(),arr.max(),ab.mean(),ab.min(),ab.max())
    size = img.size
    inp_ab = np.zeros((2,)+size)
    inp_mask = np.zeros((1,)+size)
    for pos,col,r in points:
        col = convRGB(col)[1:]
        col_centres = findABCentroids(col_dist,pos[0],pos[1])
        dist_2 = np.sum((col_centres - col)**2, axis=1)
        colMod = col_centres[np.argmin(dist_2)]
        put_point(inp_ab,inp_mask,pos,r,colMod)
        print(f"Old:New-{col}:{colMod},pos={pos},centres = {col_centres}")
    img,col_dist = run(img,inp_ab,inp_mask)
    col_dist = col_dist[0]
    print("COL DISTR SHAPE:",col_dist.shape)
    return img


