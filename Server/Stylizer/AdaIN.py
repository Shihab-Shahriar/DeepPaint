import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image



decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

encoder = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def transform(cF,sF,alpha = .6):
    cfSize = cF.size()
    sF = sF.squeeze(0).cpu().double()
    cF = cF.squeeze(0).cpu().double()

    C,W,H = cF.size(0),cF.size(1),cF.size(2)
    C1,W1,H1 = sF.size(0),sF.size(1),sF.size(2)

    cF = cF.view(C,-1)
    sF = sF.view(C,-1)
    
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean
    
    contentConv = torch.mm(cF,cF.t()).div(cF.size(1)-1) 
    contentConv += torch.eye(cF.size(0)).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)
    
    k_c = cF.size(0)
    for i in range(cF.size(0)):
        if c_e[i] < 0.00001:
            k_c = i
            break
            
    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break
            
    c_d = (c_e[0:k_c]).pow(-0.5)
    cM1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    cM2 = torch.mm(cM1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(cM2,cF)
    
    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    
    ccsF = alpha * targetFeature + (1-alpha) * cF
    
    out = ccsF.detach()
    out.resize_(cfSize)
    return out.cuda().float()


decoder = decoder.eval().cuda()
encoder = encoder.eval().cuda()

decoder.load_state_dict(torch.load("weights/AdaIN_decoder.pth"))
encoder.load_state_dict(torch.load("weights/AdaIN_encoder.pth"))
encoder = nn.Sequential(*list(encoder.children())[:31])

print("AdaIN initialized...")

def stylizeAdaIN(Ic,Is,alpha = 1.0):
    Ic = transforms.ToTensor()(Ic).unsqueeze(0).cuda()
    Is = transforms.ToTensor()(Is).unsqueeze(0).cuda()

    with torch.no_grad():
        fc = encoder(Ic)
        fs = encoder(Is)
        fx = alpha * adaptive_instance_normalization(fc,fs)
        fx += (1-alpha) * fc
        Ix = decoder(fx)

    out = Ix.squeeze(0).cpu().numpy().transpose(1,2,0)*255
    return Image.fromarray(out.clip(0,255).astype("uint8"))


