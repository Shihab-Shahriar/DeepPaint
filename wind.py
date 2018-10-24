from PIL import Image
import sys,os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from Gui.Gallery import Gallery,GalleryItem
from Gui.ColorizerGUI import ColorizeTab
from Gui.Gallery import OutputBox,ImageBox
#from Colorizer import colorize

class StyleList(QListWidget):
    def __init__(self,*args,**kwargs):
        super(Gallery,self).__init__(*args,**kwargs)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setIconSize(QSize(80,60))
        
        self.imgs = []

    def addImage(self,img):
        if type(img)==type(""): img = Image.open(img).resize((256,256))
        self.imgs.append(img)
        self.addItem(GalleryItem(img))
        
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
        return data
    
    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if item==None: return 
        m = QMenu()
        
        sv = m.addAction("Save")
        sv.triggered.connect(lambda :self.saveFile(item))
        dt = m.addAction("Delete")
        dt.triggered.connect(lambda :self.takeItem(self.row(item)))
        act = m.exec_(self.mapToGlobal(event.pos()))
        
    def saveFile(self,item):
        paths = QFileDialog.getSaveFileName(self,"Save painting")
        im = item.img
        im.save(paths[0])

class StylizerTab(QWidget):
    def __init__(self):
        super(StylizerTab,self).__init__()
        self.out = OutputBox()
        self.contentBox = ImageBox()
        self.styleBox = ImageBox()
        self.initUI()

    def initUI(self):
        mainL = QVBoxLayout()
        topL = QHBoxLayout()
        topLeftL = QVBoxLayout()

        self.setLayout(mainL)
        self.topL = 2
        
        self.layout = QVBoxLayout(self)

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
        self.stylizer = QWidget(self)
        layout = QVBoxLayout(self)
        self.pushButton1 = QPushButton("PyQt5 button")
        layout.addWidget(self.pushButton1)
        self.stylizer.setLayout(layout)
        
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
