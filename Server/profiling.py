import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from random import choice
import time

from Stylizer.algorithm import stylizeAbstract,stylizeAdaIN

def absStylePerf():
    pass

def measure():
    CONTENT_DIR = "D:/Images/Content/"
    STYLE_DIR = "D:/Images/Style/"
    x,y = [],[]
    for cont in os.listdir(CONTENT_DIR):
        for sty in os.listdir(STYLE_DIR):
            Ic = Image.open(CONTENT_DIR + cont)
            Is = Image.open(STYLE_DIR + sty)
            alpha = choice([x / 10 for x in range(1, 11)])
            start = time.perf_counter()
            out = stylizeAdaIN(Ic,Is,alpha)
            st = time.perf_counter() - start
            a,b = Ic.size
            c,d = Is.size
            x.append(a*b + c*d)
            y.append(st)
    x = np.array(x)
    y = np.array(y)
    plt.plot(x,y)
    plt.show()

measure()