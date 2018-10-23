from PIL import Image

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from Gui.Gallery import OutputBox

class ImageHint(QWidget): 

    updated = pyqtSignal(Image.Image)
    
    def __init__(self):
        super(ImageHint,self).__init__()
        self.chosen_points = []     
        self.brushCol = QColor(0,0,255)
        self.px = None
        self.img = None
        self.r = 5
        self.setAcceptDrops(True)

    def paintEvent(self,event):
        painter = QPainter(self)
        if self.px==None:
            painter.drawText(QPoint(self.width()//2-5,self.height()//2-5),"No Painting")
            return
        painter.drawPixmap(self.rect(), self.px)
        painter.setRenderHint(QPainter.Antialiasing, True)
        for pos,col,r in self.chosen_points:
            painter.setBrush(QBrush(col))
            qs = self.size()
            sx,sy = qs.width(),qs.height()
            painter.drawEllipse(QPoint(pos[0]*sx,pos[1]*sy),r,r)

        

    def mouseReleaseEvent(self, cursor_event):
        #print("hello:",cursor_event.pos())
        qs = self.size()
        sx,sy = qs.width(),qs.height()
        cur = cursor_event.pos()
        x,y = cur.x(),cur.y()
        rat = (x/sx,y/sy)
        self.chosen_points.append((rat,self.brushCol,self.r))
        print("Brush Col:",type(self.brushCol))
        self.update()
        self.updated.emit(self.img)
        
    def reset(self):
        #print("Hints reset")
        self.chosen_points = []
        self.update()
        self.updated.emit(self.img)
        
    def undo(self):
        #print("Last hint undone")
        self.chosen_points.pop()
        self.update()
        self.updated.emit(self.img)

    def updateRad(self,r):
        self.r = r
        
    def sizeHint(self):
        return QSize(300,300)
    
    def dragEnterEvent(self, e):
        if hasattr(e.mimeData(),'img'): e.accept()
        else: e.ignore()

    def dropEvent(self, e):
        self.img = e.mimeData().img
        print(type(self.img))
        self.px = self.img.toqpixmap()
        self.chosen_points = []
        self.update()
        self.updated.emit(self.img)


class ColorizeTab(QWidget):

    colFunc = None

    def __init__(self,parent,colorizer):
        super(ColorizeTab,self).__init__(parent)
        self.hint = ImageHint()
        self.out = OutputBox(Image.open("d:/Images/Content/stata.jpg"))
        ColorizeTab.colFunc = colorizer
        self.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.initUI()
    
    def initUI(self):
        mainL = QVBoxLayout()
        self.setLayout(mainL)
        
        self.selCol = QPushButton("select color",self)
        self.selCol.clicked.connect(self.colorChoose)
        
        self.undoBtn = QPushButton("Undo",self)     
        self.undoBtn.clicked.connect(self.hint.undo)

        self.resetBtn = QPushButton("Reset")
        self.resetBtn.clicked.connect(self.hint.reset)
        
        self.curCol = QLabel("CURRENT COLOR",self)
        self.curCol.setAlignment(Qt.AlignCenter)
        self.curCol.setStyleSheet(f"background-color: rgb(0,0,255)")
        self.curCol.setFixedSize(self.selCol.size())

        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(1)
        self.sl.setMaximum(10)
        self.sl.setValue(5)
        self.sl.setTickPosition(QSlider.TicksBelow)
        self.sl.setTickInterval(1)
        self.sl.valueChanged.connect(self.hint.updateRad)
        
        optL = QHBoxLayout()
        optL.addStretch(1)
        optL.addWidget(self.selCol)
        optL.addStretch(1)
        optL.addWidget(self.curCol)
        optL.addStretch(1)
        optL.addWidget(self.undoBtn)
        optL.addStretch(1)
        optL.addWidget(self.resetBtn)
        optL.addStretch(1)
        optL.addWidget(self.sl)
        optL.addStretch(1)
        
        self.hint.updated.connect(self.colorize)
        
        topL = QHBoxLayout()
        topL.addWidget(self.hint)
        topL.addWidget(self.out)
        
        mainL.addLayout(topL)
        mainL.addLayout(optL)
        
    def colorize(self,img):
        print("Colorizing...",type(img))
        points = []
        qs = img.size
        sx, sy = img.size[0], img.size[1]
        for pos,col,r in self.hint.chosen_points:
            print("In colorize",type(col))
            pos = (int(pos[1] * sy),int(pos[0] * sx))
            col = (col.red(),col.green(),col.blue())
            points.append((pos,col,r))

        self.out.img = ColorizeTab.colFunc(img,points)
        self.out.px = self.out.img.toqpixmap()
        self.out.update()
        print("Done")
        
    def colorChoose(self):
        colDialog = QColorDialog()
        col = colDialog.getColor()
        self.hint.brushCol = col
        print("DIALOGG:",type(col))
        self.curCol.setStyleSheet(f"background-color: rgb({col.red()},{col.green()},{col.blue()})")
        self.update()
        
        