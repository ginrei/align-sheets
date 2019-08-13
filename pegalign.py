from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob

def median_filter_1d(image, ksize):
    d = int((ksize-1)/2)
    w = len(image)

    filtered_image = image.copy()
    for x in range(d, w-d):
        filtered_image[x] = np.median(image[x-d:x+d+1])

    return filtered_image

class Sheet:
    filepath = ""
    image = np.ndarray([])
    gray  = np.ndarray([])
    center_peg = ()
    angle = 0

    def __init__(self, filepath):
        self.filepath = filepath
        self.image = np.asarray(Image.open(filepath))
        # crop 1/5 from top
        crop  = self.image[0:int(self.image.shape[0]/5), 0:self.image.shape[1]]
        self.gray  = cv.cvtColor(crop,cv.COLOR_BGR2GRAY)

        self.detectAngle()
        self.detectCenterPeg()

    def detectAngle(self):
        # generate X-ward / Y-ward gradient map via Sobel Filter
        sobel_x = cv.Sobel(self.gray,cv.CV_64F,1,0,ksize=5).flatten()
        sobel_y = cv.Sobel(self.gray,cv.CV_64F,0,1,ksize=5).flatten()

        phis = np.arange(-5, 5, 0.05)
        orientations = np.rad2deg(np.arctan(sobel_y[np.nonzero(sobel_x)] /  sobel_x[np.nonzero(sobel_x)]))
        orientations = orientations[np.nonzero(orientations)]

        # 水平方向変位成分は垂直方向変位として捉え、採用する
        for i,orientation in enumerate(orientations):
            if(orientation > 45):
                orientations[i] = orientation - 90
            elif(orientation < -45):
                orientations[i] = orientation + 90

        histogram = np.histogram(orientations, bins=phis, density=True)
        # smooth peaks of histogram with 1D Median Filter
        histogram = (median_filter_1d(histogram[0], ksize=5),histogram[1])

        # if np.amax(histogram[0]) >= 0.2:
        #     # ヒストグラムの最大値を持つ区間群を抽出する
        #     histogram_max_angles = histogram[1][:-1][histogram[0] == np.amax(histogram[0])]
        #     # もっとも原典に近いものを採用 TODO: meanをとるほうがよくないか
        #     self.angle = histogram_max_angles[abs(histogram_max_angles).argmin()]
        #
        # else:
        #     self.angle = 0

        if np.amax(histogram[0]) >= 0.2:
            # ヒストグラムの最大値を持つ区間群を抽出する
            histogram_max_angles = histogram[1][:-1][histogram[0] >= np.amax(histogram[0]) - 0.02]
            # もっとも原点に近いものを採用
            self.angle = histogram_max_angles[abs(histogram_max_angles).argmin()]
        else:
            self.angle = 0

    def detectCenterPeg(self):
        blur = cv.GaussianBlur(self.gray, (9,9), 2, 2)
        edge = cv.Canny(blur,50,150,apertureSize = 3)
        circles = cv.HoughCircles(edge, cv.HOUGH_GRADIENT, dp=1, minDist=200, param1=20, param2=20, maxRadius=int(edge.shape[0]/5))

        for i, circle in enumerate(circles[0]):
            x,y,rad = circle
            if(i == 0):
                self.center_peg = (x,y,rad)
            elif np.abs(self.gray.shape[1] / 2 - x) < np.abs(self.gray.shape[1] / 2 - self.center_peg[0]) :
                self.center_peg = (x,y,rad)

    def alignCenter(self, new_center_peg):
        peg_x,peg_y,_ = self.center_peg
        new_peg_x,new_peg_y,_ = new_center_peg
        translate_x = new_peg_x - peg_x
        translate_y = new_peg_y - peg_y
        translate_matrix = np.float32([[1,0,translate_x],[0,1,translate_y]])
        self.image = cv.warpAffine(self.image, translate_matrix, (self.image.shape[1],self.image.shape[0]))
        self.center_peg = new_center_peg

    def alignRotation(self):
        x,y,_ = self.center_peg
        # rotate around center peg
        rotation_matrix = cv.getRotationMatrix2D((x,y), self.angle, 1)
        print(self.filepath, self.angle)
        self.image = cv.warpAffine(self.image, rotation_matrix, (self.image.shape[1],self.image.shape[0]), flags=cv.INTER_CUBIC)

    def getCenterPeg(self):
        return self.center_peg

    def saveImage(self):
        filepath, filename = os.path.split(self.filepath)
        Image.fromarray(self.image).save(filepath + "/mod_" + filename)


sheets = []
center_pegs = np.array([])

for filepath in glob.glob('src/*'):
    sheets.append(Sheet(filepath))

center_peg_x = np.mean([sheet.getCenterPeg()[0] for sheet in sheets])
center_peg_y = np.mean([sheet.getCenterPeg()[1] for sheet in sheets])
center_peg_rad = np.mean([sheet.getCenterPeg()[1] for sheet in sheets])
center_peg = (center_peg_x, center_peg_y, center_peg_rad)

for sheet in sheets:
    sheet.alignCenter(center_peg)
    sheet.alignRotation()
    sheet.saveImage()
