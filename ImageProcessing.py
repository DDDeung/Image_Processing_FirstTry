from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene
import cv2
import numpy as np
from matplotlib import pyplot as plt
from function import hist2,sp_noise,gauss_noise,mean_filter,media_filter,laplacian_filter,Prewitt_sharpen
from QRCode import QRCodeDetector
from mainWindow import Ui_mainWindow

class ImageProcessing(QtWidgets.QMainWindow,Ui_mainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.menubar.setNativeMenuBar(False)
        self.zoomscale = 1
        self.scene = QGraphicsScene()  # 创建场景
        self.img_matrix=np.zeros((1,1,1))#初始图片的矩阵
        self.cur_matrix=np.zeros((1,1,1))#变换后图片的矩阵
        self.cb_matrix=np.zeros((1,1,1))

        self.light_slider.setValue(50)#亮度进度条初值
        self.contrast_slider.setValue(50)#对比度进度条初值
        self.sat_slider.setValue(50)#饱和度进度条初值
        """
        连接信号和槽
        """
        self.open_file.triggered.connect(self.openimage)
        self.save_file.triggered.connect(self.saveimage)
        self.fangda.clicked.connect(self.zoom_in)
        self.suoxiao.clicked.connect(self.zoom_out)
        self.jingxiang.clicked.connect(self.filp_image)
        self.xuanzhuan.clicked.connect(self.rotate)
        self.color_hist.triggered.connect(self.show_hist)
        self.gray_hist.triggered.connect(self.show_gray_hist)
        self.light_slider.sliderReleased.connect(self.bright_value_change)#释放进度条
        self.contrast_slider.sliderReleased.connect(self.contrast_value_change)
        self.sat_slider.sliderReleased.connect(self.sat_value_change)
        self.color_change_gray.clicked.connect(self.color2gray)
        self.binary.clicked.connect(self.mean_binarization)
        self.recover.clicked.connect(self.originate)
        self.rgb_hist.clicked.connect(self.hist_image)
        self.sp_noise.clicked.connect(self.add_sp_noise)
        self.guass_noise.clicked.connect(self.add_gauss_noise)
        self.mean_filter.clicked.connect(self.rm_noise_mean)
        self.media_filter.clicked.connect(self.rm_noise_media)
        self.gauss_filter.clicked.connect(self.rm_noise_guass)
        self.QRcode_pos.clicked.connect(self.find_QRcode)
        self.Prewitt.clicked.connect(self.pre_sharpen)
        self.Canny.clicked.connect(self.canny_sharpen)

    def set_image(self,matrix):
        self.scene.clear()  # 清除图元
        x = matrix.shape[1]  # 获取图像大小
        y = matrix.shape[0]
        frame = QImage(matrix, x, y, x * 3, QImage.Format_RGB888)  # 其实不是很懂
        pix = QPixmap.fromImage(frame)
        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
        # self.item.setScale(self.zoomscale)
        self.scene.addItem(self.item)
        self.image_show.setScene(self.scene)  # 将场景添加至视图



    def openimage(self):
        imgName,imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        if imgName is None:
            return
        else:
            img = cv2.imread(imgName)  # 读取图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道 OpenCV中是以BGR的顺序排列，Qt中是以RGB的顺序排列
            self.img_matrix=img;self.cur_matrix=img;self.cb_matrix=img
            self.set_image(self.cur_matrix)


    def saveimage(self):
        filename,filetype=QFileDialog.getSaveFileName(self,"保存图片","",'‘Image files(*.jpg *.png)')
        if filename is None:
            return
        else:
            # rect = self.image_show.scene().sceneRect()
            # pixmap = QtGui.QImage(rect.height(), rect.width(), QtGui.QImage.Format_ARGB32_Premultiplied)
            # painter = QtGui.QPainter(pixmap)
            # rectf = QRectF(0, 0, pixmap.rect().height(), pixmap.rect().width())
            # self.image_show.scene().render(painter, rectf, rect)
            pixmap=self.image_show.grab()
            pixmap.save(filename)
            # pass

    def zoom_in(self):
        """
        放大图片
        """
        self.zoomscale = self.zoomscale + 0.05
        if self.zoomscale >= 1.2:
            self.zoomscale = 1.2
        cur_matrix=cv2.resize(self.img_matrix,(0,0),fx=self.zoomscale,fy=self.zoomscale, interpolation=cv2.INTER_NEAREST)
        self.cur_matrix=cur_matrix;self.cb_matrix=cur_matrix
        self.set_image(self.cur_matrix)
        pass


    def zoom_out(self):
        """
        缩小图片
        """
        self.zoomscale = self.zoomscale - 0.05
        if self.zoomscale <= 0:
            self.zoomscale = 0.2

        cur_matrix=cv2.resize(self.img_matrix,(0,0),fx=self.zoomscale,fy=self.zoomscale, interpolation=cv2.INTER_NEAREST)
        self.cur_matrix = cur_matrix;self.cb_matrix=cur_matrix
        self.set_image(self.cur_matrix)
        pass

    def filp_image(self):
        self.cur_matrix=cv2.flip(self.cur_matrix,1)
        self.img_matrix = cv2.flip(self.img_matrix, 1)#对初始图片也镜像
        self.cb_matrix=self.cur_matrix
        self.set_image(self.cur_matrix)

    def rotate(self):
        cols,rows,dim=self.cur_matrix.shape
        # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        # self.cur_matrix = cv2.warpAffine(self.cur_matrix, M, (cols, rows))
        trans_img = cv2.transpose(self.cur_matrix)#先转置
        self.cur_matrix = cv2.flip(trans_img, 1)#再镜像
        #对初始图片也旋转
        trans_ori_img=cv2.transpose(self.img_matrix)
        self.img_matrix=cv2.flip(trans_ori_img,1)
        #处理cb_matrix
        self.cb_matrix=self.cur_matrix

        self.set_image(self.cur_matrix)
        # print(self.cur_matrix)

    def show_hist(self):
        hist_image1 = cv2.calcHist([self.cur_matrix], [0], None, [256], [0, 256])
        hist_image2 = cv2.calcHist([self.cur_matrix], [1], None, [256], [0, 256])
        hist_image3 = cv2.calcHist([self.cur_matrix], [2], None, [256], [0, 256])
        plt.figure("RGB分布直方图",figsize=(10,6))
        plt.subplot(131)
        plt.plot(hist_image1,'r')
        plt.title('R')
        plt.subplot(132)
        plt.plot(hist_image2,'g')
        plt.title('G')
        plt.subplot(133)
        plt.plot(hist_image3,'b')
        plt.title('B')
        plt.show()

    def show_gray_hist(self):
        hist_image = cv2.calcHist([self.cur_matrix], [0], None, [256], [0, 256])
        plt.figure("灰度分布直方图", figsize=(5, 5))
        plt.plot(hist_image,'black')
        plt.show()


    def bright_value_change(self):
        light_value=self.light_slider.value()
        light_change=light_value-50
        # print(light_change)
        h = self.cb_matrix.shape[0]
        w=self.cb_matrix.shape[1]
        bimg=np.ones((h,w,3),dtype=np.uint8)
        for i in range(0,h):
            for j in range(0,w):
                for c in range(3):
                    color = int(self.cb_matrix[i,j][c] + light_change)
                    if color>255:color=255
                    elif color<0:color=0
                    bimg[i,j][c]=color
        # self.bright_matrix=bimg
        self.cur_matrix=bimg

        self.set_image(self.cur_matrix)
        pass

    def contrast_value_change(self):
        contrast_value=self.contrast_slider.value()
        contrast_scale=(contrast_value-50)/50
        h = self.cur_matrix.shape[0]
        w = self.cur_matrix.shape[1]
        cimg = np.ones((h, w, 3), dtype=np.uint8)
        for i in range(0, h):
            for j in range(0, w):
                for c in range(3):
                    color = int(self.cur_matrix[i, j][c] *(1+contrast_scale))#<1 减小 >1 增强
                    if color > 255:
                        color = 255
                    elif color < 0:
                        color = 0
                    cimg[i, j][c] = color
        # self.bright_matrix=bimg
        self.cur_matrix = cimg
        self.cb_matrix=cimg
        self.set_image(self.cur_matrix)
        pass

    def sat_value_change(self):
        simg=self.cb_matrix.astype(np.float32)
        sat_value=self.sat_slider.value()-50
        MAX_VALUE=50
        hls_img=cv2.cvtColor(simg,cv2.COLOR_RGB2HLS)
        hlsCopy = np.copy(hls_img)
        # HLS空间通道2是饱和度，对饱和度进行线性变换，且最大值在255以内，这一归一化了，所以应在1以内
        hlsCopy[:, :, 2] = (1.0 + sat_value / MAX_VALUE) * hlsCopy[:, :, 2]
        hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
        hls_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2RGB)
        # print(hls_img)
        self.cur_matrix = np.uint8(hls_img)
        self.set_image(self.cur_matrix)
        pass

    def color2gray(self):
        #获得rgb图像
        r = self.cur_matrix[:, :, 0]
        g = self.cur_matrix[:, :, 1]
        b = self.cur_matrix[:, :, 2]
        h = self.cur_matrix.shape[0]
        w = self.cur_matrix.shape[1]
        gray_image = np.ones((h, w, 3), dtype=np.uint8)
        for i in range(0,h):
            for j in range(0,w):
                value=int(max(r[i,j],g[i,j],b[i,j]))
                for c in range(3):
                    gray_image[i, j][c] = value
        # print(gray_image)
        self.cur_matrix=gray_image
        self.set_image(self.cur_matrix)

    def mean_binarization(self):#均值二值化
        img_gray = self.cur_matrix[:, :, 0]
        threshold = np.mean(img_gray)
        for c in range(3):
            self.cur_matrix[:,:,c][img_gray > threshold] = 255
            self.cur_matrix[:,:,c][img_gray <= threshold] = 0
        self.set_image(self.cur_matrix)
        # print(self.cur_matrix)

    def originate(self):
        self.cur_matrix=self.img_matrix
        self.zoomscale=1
        self.set_image(self.img_matrix)

    def hist_image(self):
        hist_image=hist2(self.cur_matrix)
        # 彩色图像均衡化,需要分解通道 对每一个通道均衡化 调库
        # (b, g, r) = cv2.split(self.cur_matrix)
        # bH = cv2.equalizeHist(b)
        # gH = cv2.equalizeHist(g)
        # rH = cv2.equalizeHist(r)
        # # 合并每一个通道
        # hist_image= cv2.merge((bH, gH, rH))
        self.cur_matrix=hist_image
        self.set_image(self.cur_matrix)

    def add_sp_noise(self):
        out_put=sp_noise(self.cur_matrix,0.1)
        self.cur_matrix=out_put
        self.set_image(self.cur_matrix)

    def add_gauss_noise(self):
        out_put=gauss_noise(self.cur_matrix,0,0.0001)
        self.cur_matrix = out_put
        self.set_image(self.cur_matrix)

#均值滤波
    def rm_noise_mean(self):
        # result = cv2.blur(self.cur_matrix, (5, 5))  # 可以更改核的大小 调库
        # self.cur_matrix=result
        # self.set_image(self.cur_matrix)
        filter=np.ones((5,5,3))
        out=mean_filter(self.cur_matrix,filter)
        self.cur_matrix=out
        self.set_image(self.cur_matrix)

        pass

    def rm_noise_media(self):
        # result = cv2.medianBlur(self.cur_matrix, 7)  # 可以更改核的大小 调库
        # self.cur_matrix = result
        # self.set_image(self.cur_matrix)
        filter = np.ones((5, 5, 3))
        out = media_filter(self.cur_matrix, filter)
        self.cur_matrix = out
        self.set_image(self.cur_matrix)

    def rm_noise_guass(self):
        result=cv2.GaussianBlur(self.cur_matrix,(3,3),0)
        self.cur_matrix=result
        self.set_image(self.cur_matrix)
        pass

    def myROI(self):
        # ROI=np.zeros(self.cur_matrix.shape,np.uint8)#保存ROI信息
        # gray = cv2.cvtColor(self.cur_matrix, cv2.COLOR_BGR2GRAY)  # 灰度化
        # ret, binary = cv2.threshold(gray,
        #                            0, 255,
        #                            cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)  # 自适应二值化
        #
        # contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # 查找所有轮廓，每个轮廓信息保存于contours数组中
        #
        # for cnt in range(len(contours)):  # 基于轮廓数量处理每个轮廓
        #     # 轮廓逼近，具体原理还需要深入研究
        #     epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
        #     approx = cv2.approxPolyDP(contours[cnt], epsilon, True)  # 保存逼近结果的顶点信息
        #     # 顶点个数决定轮廓形状
        #     # 计算轮廓中心位置
        #     mm = cv2.moments(contours[cnt])
        #     if mm['m00'] != 0:
        #         cx = int(mm['m10'] / mm['m00'])
        #         cy = int(mm['m01'] / mm['m00'])
        #         color = self.cur_matrix[cy][cx]
        #         color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
        #         p = cv2.arcLength(contours[cnt], True)
        #         area = cv2.contourArea(contours[cnt])
        #
        #         # 分析几何形状
        #         corners = len(approx)
        #         if corners == 3 and (color[2] >= 150 or color[0] >= 150) and area > 1000:  # 一系列判定条件是由该项目的特点所调整的
        #             cv2.drawContours(ROI, contours, cnt, (255, 255, 255),
        #                             -1)  # 在ROI空画布上画出轮廓，并填充白色（最后的参数为轮廓线条宽度，如果为负数则直接填充区域）
        #             imgroi = ROI & self.cur_matrix  # ROI和原图进行与运算，筛出原图中的ROI区域
        #             cv2.imshow("ROI", imgroi)
        #
        #
        #         if corners >= 10 and (color[2] >= 150 or color[0] >= 150) and area > 1000:
        #             cv2.drawContours(ROI, contours, cnt, (255, 255, 255), -1)
        #             imgroi = ROI & self.cur_matrix
        #             cv2.imshow("ROI", imgroi)
        img=self.cur_matrix.copy()
        gray = cv2.cvtColor(self.cur_matrix, cv2.COLOR_BGR2GRAY)  # 灰度化
        ret, binary = cv2.threshold(gray,
                                   0, 255,
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)  # 自适应二值化

        contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # 查找所有轮廓，每个轮廓信息保存于contours数组中
        roi=cv2.drawContours(img, contours, -1,(0,255,0),1)
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        pass

    def find_QRcode(self):
        obj=QRCodeDetector()
        image=obj.detect(self.cur_matrix)
        cv2.imshow('QRCODE', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pass

    def pre_sharpen(self):
        Prewitt_sharpen(self.cur_matrix)
        pass

    def canny_sharpen(self):
        img=cv2.Canny(self.cur_matrix[:,:,0],50, 150)
        cv2.imshow('Canny', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
