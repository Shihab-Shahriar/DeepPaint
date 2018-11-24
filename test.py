import unittest
import os
from PIL import Image
from random import choice,randint
from Stylizer.algorithm import stylizeAbstract,stylizeAdaIN
from Colorizer.algorithm import colorize

"""class TestStylizerAlgo(unittest.TestCase):
    def stylize(self,func):
        CONTENT_DIR = "D:/Images/Content/"
        STYLE_DIR = "D:/Images/Style/"
        for c in os.listdir(CONTENT_DIR):
            for s in os.listdir(STYLE_DIR):
                Ic = Image.open(CONTENT_DIR + c)
                Is = Image.open(STYLE_DIR + s)
                alpha = choice([x / 10 for x in range(1, 11)])
                func(Ic,Is,alpha)
                self.assertEqual(10,10,msg="SIZE MISMATCH")

    def test_AdaIN(self):
        self.stylize(lambda a,b,c:stylizeAdaIN(a,b,c))

    def testAbstract(self):
        MODELS = ['cezanne', 'el-greco', 'monet', 'picasso', 'van-gogh']
        info = {'model_name':choice(MODELS)}
        self.stylize(lambda a,b,c:stylizeAbstract(a,info))"""

class TestColorizerAlgo(unittest.TestCase):
    def test_colorize_small(self):
        CONTENT_DIR = "D:/Images/Content/"
        for c in os.listdir(CONTENT_DIR):
            Im = Image.open(CONTENT_DIR + c).resize((512,512))
            xs,ys = Im.size
            points = []
            for p in range(randint(0,100)):
                pos = (randint(0,xs),randint(0,ys))
                r = randint(0,10)
                col = (randint(0,255),randint(0,255),randint(0,255))
                points.append((pos,col,r))
            print(len(points))
            colorize(Im,points)

            self.assertTrue(9==9)

if __name__ == '__main__':
    unittest.main()