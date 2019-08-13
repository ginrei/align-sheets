from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2 as cv
import os
import glob


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
        crop  = self.image[0:150, 170:370] # self.image[0:int(self.image.shape[0]/5), 0:self.image.shape[1]]
        self.gray  = cv.cvtColor(crop,cv.COLOR_BGR2GRAY)
        self.gray  = cv.GaussianBlur(self.gray, (9,9), 2, 2)

    def transform(self, warp_matrix):
        self.gray = cv.warpAffine(self.gray, warp_matrix, (self.gray.shape[1],self.gray.shape[0]), flags=cv.INTER_CUBIC)
        self.image = cv.warpAffine(self.image, warp_matrix, (self.image.shape[1],self.image.shape[0]), flags=cv.INTER_CUBIC)

    def getGrayImage(self):
        return self.gray;

    def saveImage(self):
        filepath, filename = os.path.split(self.filepath)
        Image.fromarray(self.image).save(filepath + "/mod_" + filename)


sheets = []
center_pegs = np.array([])

warp_mode = cv.MOTION_EUCLIDEAN
warp_matrix = np.eye(2, 3, dtype=np.float32)

number_of_iterations = 5000;
termination_eps = 1e-10;

criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

for filepath in glob.glob('src/*'):
    sheets.append(Sheet(filepath))

for i, sheet in enumerate(sheets[1:]):
    prev_sheet = sheets[0]
    cc, warp_matrix = cv.findTransformECC (prev_sheet.getGrayImage(),sheet.getGrayImage(),warp_matrix, warp_mode, criteria)
    print(warp_matrix)
    sheet.transform(warp_matrix)
    sheet.saveImage()
    if i == 1:
        prev_sheet.saveImage()
