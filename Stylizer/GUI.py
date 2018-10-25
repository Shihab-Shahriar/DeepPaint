from PIL import Image
import sys,os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Gallery import GalleryItem
from Gallery import OutputBox,ImageBox
from Stylizer.algorithm import stylize

class StyleList(QListWidget):
    def __init__(self,*args,**kwargs):
        super(StyleList,self).__init__(*args,**kwargs)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setIconSize(QSize(80,60))

        self.setFlow(QListView.LeftToRight)

        pdir = "Pics/"
        models = ['cezanne','el-greco','monet','picasso','van-gogh']
        paths = [pdir+f+".jpg" for f in models]
        for path,model_name in zip(paths,models):
            self.addImage(path,model_name)

    def addImage(self,img,info=None):
        if type(img)==type(""): 
            img = Image.open(img).resize((256,256))
        item = GalleryItem(img)
        item.info = info
        self.addItem(item)
        
    def dragEnterEvent(self, event):
        print("Drag entring in Gallery...")
        event.accept()
        
    def dragMoveEvent(self,e):
        e.accept()
        
    def dropEvent(self,e):
        img = e.mimeData().img
        self.addImage(img)
        
    def mimeData(self,item):
        data = QMimeData()
        data.img = item[0].img
        data.info = item[0].info
        return data

class StylizerTab(QWidget):
    def __init__(self,parent):
        super(StylizerTab,self).__init__(parent)
        self.out = OutputBox()
        self.contentBox = ImageBox()
        self.styleBox = ImageBox()
        self.styles = StyleList(self)
        self.contentBox.updated.connect(self.stylizeFunc)
        self.styleBox.updated.connect(self.stylizeFunc)
        self.initUI()

    def initUI(self):
        mainL = QVBoxLayout()
        topL = QHBoxLayout()
        topLeftL = QVBoxLayout()
        self.setLayout(mainL)
        
        topLeftL.addWidget(self.contentBox)
        topLeftL.addWidget(self.styleBox)

        topL.addLayout(topLeftL)
        topL.addWidget(self.out)
        
        
        bottomL = QVBoxLayout(self)
        bottomL.addWidget(self.styles)

        mainL.addLayout(topL)
        mainL.addLayout(bottomL)

    def stylizeFunc(self,img):
        if self.styleBox.img==None or self.contentBox.img==None:
            return
        if self.styleBox.info==None:
            self.out.img = stylize(self.contentBox.img,self.styleBox.img,None)
        else:
            self.out.img = stylize(self.contentBox.img,self.styleBox.img,self.styleBox.info) 

        self.out.px = self.out.img.toqpixmap()
        self.out.update()