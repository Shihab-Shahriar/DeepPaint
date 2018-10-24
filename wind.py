from PIL import Image
import sys,os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Gui.Gallery import Gallery,GalleryItem
from Gui.Gallery import OutputBox,ImageBox
from Gui.stylizerGUI import StylizerTab
from Gui.ColorizerGUI import ColorizeTab
#from Colorizer import colorize



class Window(QWidget):
    def __init__(self):
        super(Window,self).__init__()
        self.layout = QVBoxLayout(self)
        
        #-----------------bottom part : Begin---------------
        btmlayout = QHBoxLayout(self)
        
        self.gallery = Gallery(self)
        pdir = "Pics/"
        paths = [pdir+f for f in os.listdir(pdir) if f.endswith(".jpg") or f.endswith(".png")]
        for p in paths:
            self.gallery.addImage(p)
        self.gallery.setMaximumWidth(200)
        self.gallery.setDragEnabled(True)
        
        self.colorizer = ColorizeTab(self,lambda x,y:x)
        self.stylizer = StylizerTab(self,len)
        
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.colorizer,"Colorizer")
        self.tabs.addTab(self.stylizer,"Stylizer")
        
        self.gallery.setSizePolicy(self.tabs.sizePolicy())
        btmlayout.addWidget(self.gallery)
        btmlayout.addWidget(self.tabs)
        
        #--------------bottom part : End---------------
        
        self.title = QLabel("Deep Paint",self)
        self.title.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.title)
        self.layout.addLayout(btmlayout)
        self.setGeometry(230,250,800,600)

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance() 
    w = Window()
    w.show()
    sys.exit(app.exec_())
