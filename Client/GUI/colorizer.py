from PIL import Image

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from Gallery import OutputBox,ImageBox
from remote import remote_call

class ImageHint(ImageBox): 

    updated = pyqtSignal(Image.Image)
    
    def __init__(self):
        super(ImageHint,self).__init__()
        self.chosen_points = []     
        self.brushCol = QColor(0,0,255)
        self.r = 5

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
        if self.chosen_points:
            self.chosen_points.pop()
            self.update()
            self.updated.emit(self.img)

    def updateRad(self,r):
        self.r = r
        
    def sizeHint(self):
        return QSize(300,300)

    def dropEvent(self, e): 
        self.img = e.mimeData().img
        print(type(self.img))
        self.px = self.img.convert("L").toqpixmap()
        self.chosen_points = []
        self.update()
        self.updated.emit(self.img)


class ColorizeTab(QWidget):

    def __init__(self,parent):
        super(ColorizeTab,self).__init__(parent)
        self.hint = ImageHint()
        self.out = OutputBox()
        self.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))
        self.hint.updated.connect(self.colorizeFunc)

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
        
        
        
        topL = QHBoxLayout()
        topL.addWidget(self.hint)
        topL.addWidget(self.out)
        
        mainL.addLayout(topL)
        mainL.addLayout(optL)
        
    def colorizeFunc(self,img):
        #print("Colorizing...",type(img))
        orig_size = img.size
        img = img.resize((512,512))
        points = []
        qs = img.size
        sx, sy = img.size[0], img.size[1]
        for pos,col,r in self.hint.chosen_points:
            #print("In colorize",type(col))
            pos = (int(pos[1] * sy),int(pos[0] * sx))
            col = (col.red(),col.green(),col.blue())
            points.append((pos,col,r))
            
        msg = {'type':'colorize','img':img,'points':points}
        self.out.img = remote_call(msg).resize(orig_size)
        self.out.px = self.out.img.toqpixmap()
        self.out.update()
        #print("Done")
        
    def colorChoose(self):
        colDialog = QColorDialog()
        col = colDialog.getColor()
        self.hint.brushCol = col
        #print("DIALOGG:",type(col))
        self.curCol.setStyleSheet(f"background-color: rgb({col.red()},{col.green()},{col.blue()})")
        self.update()
        
        