import numpy as np
import copy
import cv2


class QRCodeDetector:
    def __init__(self):
        pass

    def _detect_contours(self, image, ksize=5, blocksize = 11):
        '''提取所有轮廓'''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#图片灰度化处理
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)#高斯滤波进行平滑处理
        adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 2)#二值化
        adap = cv2.Canny(adap, 100, 200)#Canny算子进行边缘检测
        contours, hierachy = cv2.findContours(adap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#提取轮廓
        # print(hierachy.shape)
        return image, contours, hierachy


    def _compute_center(self, contours, i):
        '''计算轮廓中心点'''
        M = cv2.moments(contours[i])#计算图像矩
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy


    def _isParent(self, i,j,hierachy):
        '''
        判断轮廓i是否包含轮廓j
        contours存储轮廓之间的关系和层次。所以可以检测出二维码中的三个定位方框
        '''
        j_father = hierachy[j][3]#paraent
        while j_father >= 0:
            if j_father == i:
                return True
            else:
                j_father = hierachy[j_father][3]
        return False


    def _juge_angle(self, rec, hierachy):
        '''
        判断寻找是否有三个点可以围成等腰直角三角形
        #二维码的角
        '''
        if len(rec) < 3:
            return -1, -1, -1
        elif len(rec) == 3:
            return 0, 1, 2
        for i in range(len(rec)):
            for j in range(i + 1, len(rec)):
                if self._isParent(rec[i][2], rec[j][2] , hierachy): continue
                for k in range(j + 1, len(rec)):
                    if self._isParent(rec[j][2], rec[k][2], hierachy): continue
                    distance_1 = np.sqrt((rec[i][0] - rec[j][0]) ** 2 + (rec[i][1] - rec[j][1]) ** 2)
                    distance_2 = np.sqrt((rec[i][0] - rec[k][0]) ** 2 + (rec[i][1] - rec[k][1]) ** 2)
                    distance_3 = np.sqrt((rec[j][0] - rec[k][0]) ** 2 + (rec[j][1] - rec[k][1]) ** 2)
                    # print("distance:",distance_1,distance_2,distance_3)
                    if abs(distance_1 - distance_2) < 10:
                        if abs(np.sqrt(np.square(distance_1) + np.square(distance_2)) - distance_3) < 10:
                            return i, j, k
                    elif abs(distance_1 - distance_3) < 10:
                        if abs(np.sqrt(np.square(distance_1) + np.square(distance_3)) - distance_2) < 10:
                            return i, j, k
                    elif abs(distance_2 - distance_3) < 10:
                        if abs(np.sqrt(np.square(distance_2) + np.square(distance_3)) - distance_1) < 10:
                            return i, j, k
        return -1, -1, -1


    def _find(self, image, contours, hierachy):
        '''找到符合要求的轮廓'''
        rec = []
        for i in range(len(hierachy)):
            child = hierachy[i][2]
            level = 1
            while hierachy[child][2] >= 0:
                child = hierachy[child][2]
                level += 1
            if level >= 6:
                cx,cy = self._compute_center(contours, i)
                rec.append([cx,cy,i])
        '''计算得到所有在比例上符合要求的轮廓中心点'''
        # print(rec)
        i, j, k = self._juge_angle(rec, hierachy)
        print(i,j,k)
        if i == -1 or j == -1 or k == -1:
            return -1,-1,-1,-1
        ts = np.concatenate((contours[rec[i][2]], contours[rec[j][2]], contours[rec[k][2]]))
        rect = cv2.minAreaRect(ts)
        # print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        top, bottom, left, right = min(box[:,0]), max(box[:,0]), min(box[:,1]), max(box[:,1])
        # print(top,bottom,left,right)
        # print(box)
        result = copy.deepcopy(image)
        cv2.drawContours(result, [box], 0, (0, 0, 255), 5)
        cv2.drawContours(image, contours, rec[i][2], (255, 0, 0), 5)
        cv2.drawContours(image, contours, rec[j][2], (255, 0, 0), 5)
        cv2.drawContours(image, contours, rec[k][2], (255, 0, 0), 5)
        cv2.imshow('三个固定角', image)
        cv2.waitKey(0)
        cv2.imshow('二维码区域', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return top, bottom, left, right

    def detect(self, image):
        img = image.copy()
        for i, j in [(5, 7), (5, 9), (5, 11), (3, 5), (3, 7)]:
            img, contours, hierachy = self._detect_contours(img, i, j)
            top, bottom, left, right = self._find(img, contours, np.squeeze(hierachy))
            if -1 not in [top, bottom, left, right]:
                break
        img = img[left:right + 1, top:bottom + 1]
        return img

