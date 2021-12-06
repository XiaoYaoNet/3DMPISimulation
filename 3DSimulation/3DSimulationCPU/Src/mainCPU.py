from UI.myMainWindow import *
from PyQt5.QtWidgets import  QApplication
import appdirs
import pytools
import pytools.prefork


##  ============窗体测试程序 ================================
if  __name__ == "__main__":        #用于当前窗体测试
   app = QApplication(sys.argv)    #创建GUI应用程序
   form=QmyMainWindow()            #创建窗体
   form.show()
   sys.exit(app.exec_())