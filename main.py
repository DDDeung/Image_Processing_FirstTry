import sys
from ImageProcessing import  ImageProcessing
from PyQt5.QtWidgets import QApplication

if __name__=='__main__':
    app=QApplication(sys.argv)
    window=ImageProcessing()
    window.show()
    sys.exit(app.exec())
