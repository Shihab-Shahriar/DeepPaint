from PIL import Image
import sys,os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Gallery import *
from Stylizer.GUI import StylizerTab
from Colorizer.GUI import ColorizeTab


class Window(QWidget):
    def __init__(self):
        super(Window,self).__init__()
        self.myMenuBar = QMenuBar(self)
        file_menu = QMenu('File', self)
        file_action = QAction("Open file", self) # title and parent
        file_action.setStatusTip("Upload a new Painting to gallery")
        file_action.triggered.connect(self.uploadImage)
        file_menu.addAction(file_action)
        self.myMenuBar.addMenu(file_menu)


        self.layout = QVBoxLayout(self)
        
        #-----------------bottom part : Begin---------------
        btmlayout = QHBoxLayout(self)
        
        self.gallery = Gallery(self)
        pdir = "Pics/"
        paths = [pdir+f for f in os.listdir(pdir) if f.endswith(".jpg") or f.endswith(".png")]
        for p in paths:
            try:
                self.gallery.addImage(p)
            except:
                pass
        self.gallery.setMaximumWidth(200)
        self.gallery.setDragEnabled(True)
        
        self.colorizer = ColorizeTab(self)
        self.stylizer = StylizerTab(self)
        
        self.tabs = QTabWidget(self)
        self.tabs.addTab(self.colorizer,"Colorizer")
        self.tabs.addTab(self.stylizer,"Stylizer")
        
        self.gallery.setSizePolicy(self.tabs.sizePolicy())
        btmlayout.addWidget(self.gallery)
        btmlayout.addWidget(self.tabs)
                
        self.title = QLabel("Deep Paint",self)
        self.title.setAlignment(Qt.AlignCenter)
        
        self.layout.addWidget(self.title)
        self.layout.addLayout(btmlayout)
        self.setGeometry(230,250,800,600)

    def uploadImage(self):
        imagePath, _ = QFileDialog.getOpenFileName()
        self.gallery.addImage(imagePath)

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance() 
    w = Window()
    w.show()
    sys.exit(app.exec_())
