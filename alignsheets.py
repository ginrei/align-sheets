from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2 as cv
import time

class Timer():
    start_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print("{0:.4f}sec".format(time.time() - self.start_time))

def hough_line(img):
  # Rho and Theta ranges
  # thetas = np.deg2rad(np.arange(-90.0, 90.0))
  thetas = np.deg2rad(np.arange(-5, 5, 0.1))
  width  = img.width
  height = img.height
  diag_len = int(np.ceil(np.sqrt(width * width + height * height)))   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  angle_accumulator = np.zeros(num_thetas, dtype=np.uint64)
  y_idxs, x_idxs = np.nonzero(np.asarray(img))  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[int(rho), t_idx] += 1

  return accumulator, thetas, rhos


image = Image.open("./test3.jpg")
imageOriginal = image

image = image.crop((0, 0, image.width, image.height))
image = image.filter(ImageFilter.FIND_EDGES)
image = image.convert("L") # Convert to 8bit Grayscale

# image.show()

timer = Timer()
timer.start()

# accumulator, thetas, rhos = hough_line(image)
#
# index = np.argsort(accumulator.flatten())[::-1]
#
# for i in range(10):
#     idx = index[i]
#     rho = rhos[int(round(idx / accumulator.shape[1]))]
#     theta = thetas[idx % accumulator.shape[1]]
#     print("rho={0:.2f}, theta={1:.2f}".format(rho, np.rad2deg(theta)))

timer.stop()

timer.start()

image = np.asarray(imageOriginal)
edges = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
edges = cv.medianBlur(edges,5)
edges = cv.Canny(edges,50,150,apertureSize = 5)
Image.fromarray(edges).show()
minLineLength = 100
maxLineGap = 10

lines = cv.HoughLinesP(edges,1,np.pi/1800,100,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv.line(image,(x1,y1),(x2,y2),(0,255,0),2)

# circles = cv.HoughCircles(image,cv.HOUGH_GRADIENT,1,20,
#                             param1=50,param2=30,minRadius=0,maxRadius=0)
#
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # draw the outer circle
#     cv.circle(imageOriginal,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv.circle(imageOriginal,(i[0],i[1]),2,(0,0,255),3)

Image.fromarray(image).show()

timer.stop()
