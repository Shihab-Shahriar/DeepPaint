from PIL import Image

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ImageBox(QWidget):
    def __init__(self,px=None):
        super(ImageBox,self).__init__()
        self.px = px
    
    def paintEvent(self, paint_event):
        painter = QPainter(self)
        if self.px==None:
            painter.drawText(QPoint(self.width()//2-5,self.height()//2-5),"No Painting")
        else:
            painter.drawPixmap(self.rect(), self.px)
            painter.setRenderHint(QPainter.Antialiasing, True)
        
    def sizeHint(self):
        return QSize(300,300)
    
class OutputBox(ImageBox):
    def __init__(self,img=None):
        self.img = img
        self.px = None if img==None else img.toqpixmap()
        super(OutputBox,self).__init__(self.px)
        
    def mouseMoveEvent(self, e):

        if e.buttons() != Qt.RightButton or self.px==None:
            return

        mimeData = QMimeData()
        print("Drag in outputBox")
        mimeData.img = self.img
        drag = QDrag(self)
        drag.setMimeData(mimeData)
        drag.setHotSpot(e.pos() - self.rect().topLeft())

        dropAction = drag.exec_(Qt.MoveAction)
        
class GalleryItem(QListWidgetItem):
    def __init__(self,img):
        super(GalleryItem,self).__init__()
        self.img = img
        icon = QIcon(self.img.toqpixmap())
        self.setIcon(icon)
        
        
class Gallery(QListWidget):
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
        if type(img)==type(""): img = Image.open(img)
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
        print(qd)
        im = item.img
        im.save(paths[0])
