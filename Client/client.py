from PIL import Image
import sys
import argparse
import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Gallery import *
from GUI.stylizer import StylizerTab
from GUI.colorizer import ColorizeTab

import remote

class Window(QWidget):
    def __init__(self,pdir=None):
        super(Window,self).__init__()

        #----------------MENU-----------------------
        self.myMenuBar = QMenuBar(self)
        file_menu = QMenu('File', self)
        file_action = QAction("Open file", self) # title and parent
        file_action.setStatusTip("Upload a new Painting to gallery")
        file_action.triggered.connect(self.uploadImage)
        file_menu.addAction(file_action)
        self.myMenuBar.addMenu(file_menu)

        folder_menu = QMenu('Folder', self)
        folder_action = QAction("Open folder", self) # title and parent
        folder_action.setStatusTip("Upload all Paintings of a folder to gallery")
        folder_action.triggered.connect(self.uploadImageFolder)
        folder_menu.addAction(folder_action)
        self.myMenuBar.addMenu(folder_menu)


        self.layout = QVBoxLayout(self)
        
        #-----------------bottom part : Begin---------------
        btmlayout = QHBoxLayout(self)
        
        self.gallery = Gallery(self)
        if pdir:
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

    def uploadImageFolder(self):
        folderpath = QFileDialog.getExistingDirectory(self, "Select folder")
        print(folderpath)
        tot = 0
        for f in os.listdir(folderpath):
            if f.endswith('.jpg') or f.endswith('.png'):
                tot += 1
                if tot==100: break
                self.gallery.addImage(os.path.join(folderpath,f))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-ip',type=str, default='localhost',help='server ip address')
    parser.add_argument('-p', default=1539, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    print(remote.HOST,remote.PORT)
    remote.HOST,remote.PORT = args.ip,args.p
    print(args,remote.HOST,remote.PORT)


    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance() 
    w = Window()
    w.show()
    sys.exit(app.exec_())
