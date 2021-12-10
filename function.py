import numpy as np
import random
import tqdm
import cv2
import matplotlib.pyplot as plt

#灰度图像的直方图均衡化
def hist(Amatrix):
    M = Amatrix.shape[0]  # 获取图像大小
    N = Amatrix.shape[1]
    size_image=M*N
    c=np.zeros((1,256))

    for i in range(M):
        for j in range(N):
            c[0,Amatrix[i,j]]=c[0,Amatrix[i,j]]+1

    a=np.zeros((1,256))
    for i in range(1,256):
        a[0,i]=a[0,i-1]+c[0,i]

    for i in range(1,256):
        a[0,i]=(a[0,i]/size_image)*255

    hist_matrix=np.zeros((M,N),dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            hist_matrix[i,j]=round(a[0,Amatrix[i,j]])
    return hist_matrix

#彩色图的直方图均衡化
def hist2(rgbMatrix):
    M = rgbMatrix.shape[0]  # 获取图像大小
    N = rgbMatrix.shape[1]

    R=rgbMatrix[:,:,0]
    G = rgbMatrix[:, :, 1]
    B = rgbMatrix[:, :, 2]

    Rout=hist(R)
    Gout=hist(G)
    Bout=hist(B)
    hist_image=np.zeros((M,N,3),dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            hist_image[i,j,0]=Rout[i,j]
            hist_image[i, j, 1] = Gout[i, j]
            hist_image[i, j, 2] = Bout[i, j]
    return hist_image

#增加椒盐噪声
def sp_noise(rgbMatrix,prob=0.01):
    output=np.zeros(rgbMatrix.shape,np.uint8)#增加噪声后的图片
    thres=1-prob
    for i in range(rgbMatrix.shape[0]):
        for j in range(rgbMatrix.shape[1]):
            rdn=random.random()#返回随机生成的一个实数，它在[0,1)范围内
            if rdn<prob:#如果生成的随机数小于噪声比例则将该像素点添加黑点，即椒噪声
                output[i][j]=[0,0,0]
            elif rdn>thres:#如果生成的随机数大于（1-噪声比例）则将该像素点添加白点，即盐噪声
                output[i][j]=[255,255,255]
            else:
                output[i][j]=rgbMatrix[i][j]
    return output
    pass

#增加高斯噪声 噪声服从正态分布
def gauss_noise(rgbMatrix,mean=0,var=0.001):
    """
    :param rgbMatrix: 原始图像
    :param mean: 均值
    :param var: 方差，方差越大，噪声越大
    :return: 增加噪声后的图像
    """
    image=rgbMatrix
    image=np.array(image/255,dtype=float)#将原始图片归一化
    noise=np.random.normal(mean,var**0.5,image.shape)#生成噪声的正态分布序列
    out=image+noise#将噪声和原始图像进行相加得到加噪后的图像
    if out.min()<0:
        low_clip=-1.
    else:
        low_clip=0.
    out=np.clip(out,low_clip,1.0)#将元素的大小限制在了low_clip和1之间
    out=np.uint8(out*255)
    return out
    pass

#自定义均值滤波
def mean_filter(rgbMatrix,filter):
    M=rgbMatrix.shape[0]
    N=rgbMatrix.shape[1]
    Mf=filter.shape[0]
    Nf=filter.shape[1]
    k=int((Mf-1)/2)
    image2=np.zeros((M+2*k,N+2*k,3),dtype=np.float32)
    image_out=np.zeros((M,N,3),dtype=np.uint8)
    filter_f=filter.flatten()
    coeff=np.sum(filter_f)/3

    #填充 填充最近像素值的方法
    #内部图像填充
    for i in range(k,M+k):
        for j in range(k,N+k):
            image2[i,j]=rgbMatrix[i-k,j-k]
    #填充上下边缘
    for i in range(k):
        for j in range(N):
            image2[i,j+k]=rgbMatrix[0,j]
            image2[M+k+i,j+k]=rgbMatrix[M-1,j]
    #填充左右边缘
    for i in range(M):
        for j in range(k):
            image2[i+k,j]=rgbMatrix[i,0]
            image2[i+k,N+k+j]=rgbMatrix[i,N-1]
    #填充四个角
    for i in tqdm.tqdm(range(k)) :
        for j in tqdm.tqdm(range(k)):
            image2[i, j] = rgbMatrix[0, 0]
            image2[i, j + N + k] = rgbMatrix[0, N-1]
            image2[i + M + k, j] =rgbMatrix[M-1, 0]
            image2[i + M + k, j + N + k] = rgbMatrix[M-1, N-1]
    #滤波部分
    for i in range(k,M+k):
        for j in range(k,N+k):
            sub_image=image2[i-k:i+k+1,j-k:j+k+1]
            temp1=filter*sub_image
            for c in range(3):
                temp2=np.sum(temp1[:,:,c].flatten())/coeff
                image_out[i-k,j-k,c]=np.uint8(temp2)

    return image_out

# 中值均值滤波
def media_filter(rgbMatrix, filter):
    M = rgbMatrix.shape[0]
    N = rgbMatrix.shape[1]
    Mf = filter.shape[0]
    k = int((Mf - 1) / 2)
    image2 = np.zeros((M + 2 * k, N + 2 * k, 3), dtype=np.float32)
    image_out = np.zeros((M, N, 3), dtype=np.uint8)
    filter_f = filter.flatten()

    # 填充 填充最近像素值的方法
    # 内部图像填充
    for i in range(k, M + k):
        for j in range(k, N + k):
            image2[i, j] = rgbMatrix[i - k, j - k]
    # 填充上下边缘
    for i in range(k):
        for j in range(N):
            image2[i, j + k] = rgbMatrix[0, j]
            image2[M + k + i, j + k] = rgbMatrix[M - 1, j]
    # 填充左右边缘
    for i in range(M):
        for j in range(k):
            image2[i + k, j] = rgbMatrix[i, 0]
            image2[i + k, N + k + j] = rgbMatrix[i, N - 1]
    # 填充四个角
    for i in range(k):
        for j in range(k):
            image2[i, j] = rgbMatrix[0, 0]
            image2[i, j + N + k] = rgbMatrix[0, N - 1]
            image2[i + M + k, j] = rgbMatrix[M - 1, 0]
            image2[i + M + k, j + N + k] = rgbMatrix[M - 1, N - 1]
    # 滤波部分
    for i in range(k, M + k):
        for j in range(k, N + k):
            sub_image = image2[i - k:i + k + 1, j - k:j + k + 1]
            for c in range(3):
                temp =np.median(sub_image[:, :, c])
                image_out[i - k, j - k, c] = np.uint8(temp)
    return image_out

def rgbToGray(img):

    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    # 灰度化
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)
    return out

def laplacian_filter(img, K_size=3):
    # H, W, C = img.shape
    # gray = rgbToGray(img)
    # # zero padding
    # pad = K_size // 2
    # out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    # out[pad: pad + H, pad: pad + W] = gray.copy().astype(np.float)
    # tmp = out.copy()
    # # laplacian kernle
    # K = [[0., 1., 0.],[1., -4., 1.], [0., 1., 0.]]
    # # filtering
    # for y in range(H):
    #     for x in range(W):
    #         out[pad + y, pad + x] = np.sum(K * (tmp[y: y + K_size, x: x + K_size]))
    # out = np.clip(out, 0, 255)
    # out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    # return out
    pass

def Prewitt_sharpen(rgbMatrix):
    # 灰度化处理图像
    grayImage = cv2.cvtColor(rgbMatrix, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 阈值处理
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)## prewitt 水平方向的核
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int) ## prewitt 竖直方向的核
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    #转回原来的uint8形式
    absX = cv2.convertScaleAbs(x)#梯度
    absY = cv2.convertScaleAbs(y)#梯度
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)#是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
    afterImg=Prewitt+grayImage
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图形
    titles = [u'原始图像', u'Prewitt算子']
    images = [rgbMatrix, Prewitt,afterImg]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    # return Prewitt,grayImage

def prewitt_filter(img, K_size=3):
    gray = rgbToGray(img)
    if len(img.shape) == 3:
        H, W, C = img.shape
    # 填充0
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=float)
    out[pad: pad + H, pad: pad + W] = gray.copy().astype(float)
    tmp = out.copy()
    out_v = out.copy()
    out_h = out.copy()
    ## prewitt 水平方向的核
    Kv = [[-1., -1., -1.],[0., 0., 0.], [1., 1., 1.]]
    ## prewitt 竖直方向的核
    Kh = [[-1., 0., 1.],[-1., 0., 1.],[-1., 0., 1.]]
    # filtering
    for y in range(H):
        for x in range(W):
            out_v[pad + y, pad + x] = sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
            out_h[pad + y, pad + x] = sum(Kh * (tmp[y: y + K_size, x: x + K_size]))
    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)
    out_v = out_v[pad: pad + H, pad: pad + W].astype(np.uint8)
    out_h = out_h[pad: pad + H, pad: pad + W].astype(np.uint8)
    dst = cv2.addWeighted(out_v, 0.5, out_h, 0.5, 0)
    return  dst