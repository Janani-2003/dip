# dip
### EXP 01:
```
import cv2s
import matplotlib.pyplot as plt
img=cv2.imread("coffee.jpeg",1)
cv2.imshow("cup",img)
cv2.waitKey(0)

import cv2
image=cv2.imread("coffee.jpeg",1)
w=cv2.imwrite("cup",image)
cv2.cv2.imshow("coffee.jpeg",1)
cv2.waitKey(0)

import cv2
imgg=cv2.imread("coffee.jpeg",1)
print(imgg.shape)

import random
import cv2
A=cv2.imread("coffee.jpeg",1)
for i in range(100):
    for j in range(A.shape[1]):
        A[i][j]=[random.randint(0,255),random.randint(0,255),random.randint(0,255)]
cv2.imshow("coffee.jpeg",A)
cv2.waitKey(0)

import cv2
color_img = cv2.imread("coffee.jpeg",1)
tag = color_img[200:400,300:500]
color_img[300:500,200:400] = tag
cv2.imshow('Cut And Paste',color_img)
cv2.waitKey(0)
```
### EXP 02:
```
import cv2
img = cv2.VideoCapture(0)
while(True):
    imagee,frame = img.read()
    cv2.imshow('myimage',frame)
    if cv2.waitKey(1) == ord('c'):
        break
img.release()
cv2.destroyAllWindows()

import cv2
video = cv2.VideoCapture(0)
while (True):
    cap,frame=video.read()
    cv2.imshow('Capturing Video',frame)
    if cv2.waitKey(1) == ord('q'):
        break
video.release()

import cv2
import numpy as np
img  = cv2.VideoCapture(0)
while True:
    pic,frame = img.read()
    width = int(img.get(3))
    height = int(img.get(4))
    image = np.zeros(frame.shape, np.uint8)
    small_frame = cv2.resize(frame,(0,0),fx =0.5, fy = 0.5)
    image[:height//2, :width//2]=small_frame
    image[height//2:, :width//2]=small_frame
    image[:height//2, width//2:]=small_frame
    image[height//2:, width//2:]=small_frame
    cv2.imshow('myimage',image)
    if cv2.waitKey(1) == ord('c'):
        break
img.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
img  = cv2.VideoCapture(0)
while True:
    pic,frame = img.read()
    width = int(img.get(3))
    height = int(img.get(4))
    image = np.zeros(frame.shape, np.uint8)
    small_frame = cv2.resize(frame,(0,0),fx =0.5, fy = 0.5)
    image[:height//2, :width//2]=cv2.rotate(small_frame,cv2.ROTATE_180)
    image[height//2:, :width//2]=small_frame
    image[:height//2, width//2:]=small_frame
    image[height//2:, width//2:]=cv2.rotate(small_frame,cv2.ROTATE_180)
    cv2.imshow('myimage',image)
    if cv2.waitKey(1) == ord('c'):
        break
img.release()
cv2.destroyAllWindows()
```
### EXP 03:
```
# i) Convert BGR and RGB to HSV and GRAY

import cv2
sun_color_image = cv2.imread('cat.jpg')
cv2.imshow('Original image', sun_color_image)
hsv_image = cv2.cvtColor(sun_color_image, cv2.COLOR_BGR2HSV)
cv2.imshow('BGR2HSV' ,hsv_image )
gray_image1 = cv2.cvtColor (sun_color_image, cv2.COLOR_RGB2GRAY)
cv2.imshow('RGB2GRAY', gray_image1)
cv2.waitKey(0)
cv2. destroyAllWindows()

# ii)Convert HSV to RGB and BGR

import cv2
sun_color_image = cv2.imread('cat.jpg')
cv2.imshow('Original image', sun_color_image)
hsv_image = cv2.cvtColor(sun_color_image, cv2.COLOR_HSV2RGB)
cv2.imshow('HSV2RGB' ,hsv_image )
gray_image1 = cv2.cvtColor (sun_color_image, cv2.COLOR_HSV2BGR)
cv2.imshow('HSV2BGR', gray_image1)
cv2.waitKey(0)
cv2. destroyAllWindows()

# iii)Convert RGB and BGR to YCrCb

import cv2
sun_color_image = cv2.imread('cat.jpg')
cv2.imshow('Original image', sun_color_image)
gray_image1 = cv2.cvtColor (sun_color_image, cv2.COLOR_RGB2YCrCb)
cv2.imshow('RGB2YCrCb', gray_image1)
gray_image1 = cv2.cvtColor (sun_color_image, cv2.COLOR_BGR2YCrCb)
cv2.imshow('BGR2YCrCb', gray_image1)
cv2.waitKey(0)
cv2. destroyAllWindows()

# iv)Split and Merge RGB Image

import cv2
image = cv2.imread('cat.jpg')
blue=image[:,:,0]
green=image[:,:,1]
red=image[:,:,2]
cv2.imshow('B-Channel',blue)
cv2.imshow('G-Channel',green)
cv2.imshow('R-Channel',red)
Merged_BGR=cv2.merge((blue,green,red))
cv2.imshow('Merged BGR Image',Merged_BGR)
cv2.waitKey(0)
cv2.destoryAllWindows()

# v) Split and merge HSV Image

import cv2
image = cv2.imread('cat.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
cv2.imshow('Hue-Image',h)
cv2.imshow('Saturation-Image',s)
cv2.imshow('Gray-Image',v)
Merged_HSV = cv2.merge((h,s,v))
cv2.imshow('Merged HSV Image',Merged_HSV)
cv2.waitKey(0)
cv2.destoryAllWindows()
```
### EXP 04:
```
import cv2
import matplotlib.pyplot as plt

# Write your code to find the histogram of gray scale image and color image channels.

import cv2
import matplotlib.pyplot as plt
gray_image=cv2.imread('tj.jpg')
hist=cv2.calcHist([gray_image],[0],None,[256],[0,256])
plt.imshow(gray_image)
plt.show()
plt.figure()
plt.title("histogram")
plt.xlabel('grayscale value')
plt.ylabel('pixel count')
plt.stem(hist)
plt.show()

# Display the histogram of gray scale image and any one channel histogram from color image

import cv2
import matplotlib.pyplot as plt
color_image=cv2.imread('tj.jpg')
hist=cv2.calcHist([color_image],[0],None,[256],[0,256])
plt.imshow(color_image)
plt.show()
plt.figure()
plt.title("histogram")
plt.xlabel('colorscale value')
plt.ylabel('pixel count')
plt.stem(hist)
plt.show()

# Write the code to perform histogram equalization of the image. 

import cv2
import matplotlib.pyplot as plt
gray_image = cv2.imread('tj.jpg',0)
equ=cv2.equalizeHist(gray_image)
cv2.imshow('Gray image',gray_image)
cv2.imshow('Equalized Image',equ)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### EXP 05:
```
import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread('minion.jpg')
img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(img)
plt.show()
rows,cols,dim=img.shape

i)Image Translation

m=np.float32([[1,0,100],[0,1,200],[0,0,1]])
t_img=cv2.warpPerspective(img,m,(cols,rows))
plt.axis('off')
plt.imshow(t_img)
plt.show()

ii) Image Scaling

n=np.float32([[1.2,0,0],[0,1.2,0],[0,0,1]])
s_img=cv2.warpPerspective(img,n,(cols*2,rows*2))
plt.axis('off')
plt.imshow(s_img)

iii)Image shearing

o_x=np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
p_y=np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
sh_x=cv2.warpPerspective(img,o_x,(int(cols*1.5),int(rows*1.5)))
sh_y=cv2.warpPerspective(img,p_y,(int(cols*1.5),int(rows*1.5)))
plt.axis('off')
plt.imshow(sh_x)
plt.imshow(sh_y)
plt.show()

iv)Image Reflection

rows,cols,dim=img.shape
q_x=np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
q_y=np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
r_img=cv2.warpPerspective(img,q_x,(int(cols),int(rows)))
r_img1=cv2.warpPerspective(img,q_y,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(r_img)
plt.show()
plt.imshow(r_img1)
plt.show()

v)Image Rotation

angle=np.radians(40)
r=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
ro=cv2.warpPerspective(img,r,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(ro)
plt.show()

vi)Image Cropping

c_img = img[120:600,120:600]
plt.axis('off')
plt.imshow(c_img)
plt.show()
```
### EXP 06:
```
i) Using Averaging Filter
kernel1 = np.ones((11,11),np.float32)/121
avg_filter = cv2.filter2D(original_image,-1,kernel1)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(avg_filter)
plt.title("Filtered")
plt.axis("off")

ii) Using Weighted Averaging Filter
kernel2 = np.array([[1,2,1],[2,4,2],[1,2,1]])/16
weighted_filter = cv2.filter2D(original_image,-1,kernel2)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(weighted_filter)
plt.title("Filtered")
plt.axis("off")

iii) Using Gaussian Filter
gaussian_blur = cv2.GaussianBlur(src = original_image, ksize = (11,11), sigmaX=0, sigmaY=0)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Filtered")
plt.axis("off")

iv) Using Median Filter
median = cv2.medianBlur(src=original_image,ksize = 11)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Filtered")
plt.axis("off")

2. Sharpening Filters
i) Using Laplacian Kernal
kernel3 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_kernel = cv2.filter2D(original_image,-1,kernel3)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian_kernel)
plt.title("Filtered")
plt.axis("off")

ii) Using Laplacian Operator
laplacian_operator = cv2.Laplacian(original_image,cv2.CV_64F)
plt.figure(figsize = (9,9))
plt.subplot(1,2,1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(laplacian_operator)
plt.title("Filtered")
plt.axis("off")
```
### EXP 07:
```
# Import the packages
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image, Convert to grayscale and remove noise
img=cv2.imread ('pop.jpg')
img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.title('Original_image')
plt.imshow(img1)

gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.title('GRAY_IMAGE')
plt.imshow(gray_img,cmap = 'gray')

# SOBEL EDGE DETECTOR
img = cv2.GaussianBlur(gray_img,(3,3),0)
sobelx = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5)
sobelxy =cv2.Sobel(gray_img,cv2.CV_64F,1,1,ksize=5)

plt.figure(1)
plt.subplot(1,1,1)
plt.imshow(gray_img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# SOBELx : 

plt.subplot(1,1,1)
plt.imshow(sobelx,cmap='gray')
plt.title('sobelx')
plt.xticks([]), plt.yticks([])

# SOBELy : 

plt.subplot(1,1,1)
plt.imshow(sobely,cmap='gray')
plt.title('sobely')
plt.xticks([]), plt.yticks([])

# SOBELxy :

plt.subplot(1,1,1)
plt.imshow(sobelxy,cmap='gray')
plt.title('sobelxy')
plt.xticks([]), plt.yticks([])
plt.show()

# LAPLACIAN EDGE DETECTOR
laplacian = cv2.Laplacian(gray_img,cv2.CV_64F)
plt.imshow(laplacian,cmap='gray')
plt.title('laplacian')
plt.show()

# CANNY EDGE DETECTOR
canny_edges = cv2.Canny(gray_img, 120, 150)
plt.imshow(canny_edges,cmap='gray')
plt.title('canny_edges')
plt.show()
```
### EXP 08:
```
import numpy as np
import  cv2
import matplotlib.pyplot as plt

# Read image and convert it to grayscale image
img = cv2.imread("tom.jpg")
img = cv2.GaussianBlur(img,(3,3),0)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(img,cmap='gray')
plt.title('GRAYSCALE IMAGE')
plt.xticks([])
plt.yticks([])
plt.show()

# Find the edges in the image using canny detector and display
edge = cv2.Canny(img,50,50)
plt.imshow(edge,cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])
plt.show()

# Detect points that form a line using HoughLinesP
lines=cv2.HoughLinesP(edge,1,np.pi/180, threshold=80, minLineLength=40,maxLineGap=250)

# Draw lines on the image
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(edge,(x1,y1),(x2,y2),(255,0,255),3)

# Display the result
plt.imshow(edge)
plt.title('Hough transformation')
plt.xticks([])
plt.yticks([])
plt.show()
```
### EXP 09:
```
# Load the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the Image and convert to grayscale
image=cv2.imread("moana.jpg",1)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_gray=cv2.imread("moana.jpg",0)

# Use Global thresholding to segment the image
ret,thresh_img1=cv2.threshold(image_gray,86,255,cv2.THRESH_BINARY)
ret,thresh_img2=cv2.threshold(image_gray,86,255,cv2.THRESH_BINARY_INV)
ret,thresh_img3=cv2.threshold(image_gray,86,255,cv2.THRESH_TOZERO)
ret,thresh_img4=cv2.threshold(image_gray,130,255,cv2.THRESH_TOZERO_INV)
ret,thresh_img5=cv2.threshold(image_gray,100,255,cv2.THRESH_TRUNC)

# Use Adaptive thresholding to segment the image
thresh_img7=cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
thresh_img8=cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# Use Otsu's method to segment the image 
ret,thresh_img6=cv2.threshold(image_gray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

# Display the results
titles=["Gray Image","Threshold Image (Binary)","Threshold Image (Binary Inverse)","Threshold Image (To Zero)"
       ,"Threshold Image (To Zero-Inverse)","Threshold Image (Truncate)","Otsu","Adaptive Threshold (Mean)","Adaptive Threshold (Gaussian)"]
images=[image_gray,thresh_img1,thresh_img2,thresh_img3,thresh_img4,thresh_img5,thresh_img6,thresh_img7,thresh_img8]
for i in range(0,9):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.title(titles[i])
    plt.imshow(cv2.cvtColor(images[i],cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
```
### EXP10:
```
# Import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create the Text using cv2.putText
img1=np.zeros((90,250),dtype='uint8')
img2=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
font=cv2.FONT_HERSHEY_DUPLEX
cv2.putText(img2,'JANANI',(5,70),font,2,(218, 112, 214),3,cv2.LINE_8)
plt.imshow(img2)

# Create the structuring element
kernel=np.ones((5,5),np.uint8)
kernel1=cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))

# Erode the image
image_erode1=cv2.erode(img2,kernel1)
plt.imshow(image_erode1)
# Dilate the image
image_dilate=cv2.dilate(img2,kernel1)
plt.imshow(image_dilate)
```
### EXP 11:
```
# Import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Create the Text using cv2.putText
image = np.zeros((100,250),dtype = 'uint8')
img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
font = cv2.FONT_HERSHEY_SIMPLEX = 3
cv2.putText(img,"JANANI",(5,70),font,2,(204,0,102),5,cv2.LINE_AA)
plt.title("Original Image")
plt.imshow(img,'magma')
plt.axis('off')

# Create the structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))

# Use Opening operation
opening_image = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
plt.title("Opening")
plt.imshow(opening_image,'magma')
plt.axis('off')

# Use Closing Operation
closing_image = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
plt.title("Closing")
plt.imshow(closing_image,'magma')
plt.axis('off')
```
### EXP 12:
```
string='212221230039 JANANI'
class NodeTree(object):
  def __init__(self,left=None,right=None):
    self.left=left
    self.right=right
  def children(self):
    return(self.left,self.right)

def huffman_code_tree(node,left=True,binString=''):
  if type(node) is str:
    return {node:binString}
  (l,r)=node.children()
  d=dict()
  d.update(huffman_code_tree(l,True,binString+'0'))
  d.update(huffman_code_tree(r,False,binString+'1'))
  return d

freq = {}
for c in string:
  if c in freq:
    freq[c] += 1
  else:
    freq[c] = 1
freq=sorted(freq.items(),key=lambda x:x[1],reverse=True)
nodes=freq

while len(nodes)>1:
  (key1,c1)=nodes[-1]
  (key2,c2)=nodes[-2]
  nodes=nodes[:-2]
  node=NodeTree(key1,key2)
  nodes.append((node,c1+c2))
  nodes=sorted(nodes,key=lambda x:x[1],reverse=True)

huffmanCode=huffman_code_tree(nodes[0][0])
print('Char|Huffman code')
print('-------------------')
for(char,frequency) in freq:
  print('%-4r |%12s'% (char, huffmanCode[char]))
```
