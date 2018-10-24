from PIL import Image
import sys,os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Gui.Gallery import GalleryItem
from Gui.Gallery import OutputBox,ImageBox

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
        paths = [pdir+f for f in os.listdir(pdir) if f.endswith(".jpg") or f.endswith(".png")]
        for p in paths:
            self.addImage(p,"hoo")

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

    styleFunc = None

    def __init__(self,parent,styleFunc):
        super(StylizerTab,self).__init__(parent)
        StylizerTab.styleFunc = styleFunc
        self.out = OutputBox()
        self.contentBox = ImageBox()
        self.styleBox = ImageBox()
        self.styles = StyleList(self)
        self.contentBox.updated.connect(self.stylize)
        self.styleBox.updated.connect(self.stylize)
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

    def stylize(self,img):
        if self.styleBox.img and self.contentBox.img:
            print("STYLIZING............")
            if self.styleBox.info=="hoo":
                self.out.img = self.styleBox.img
            else:
                self.out.img = self.contentBox.img   

            self.out.px = self.out.img.toqpixmap()
            self.out.update()