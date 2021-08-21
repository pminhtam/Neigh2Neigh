import cv2
# img = cv2.imread("ic_launcher_foreground.png")
# img = cv2.resize(img,(512,512))
#
# cv2.imwrite("ic.png",img)

img = cv2.imread("/home/dell/Downloads/and/1a363c0f52faa6a4ffeb5.jpg")
print(img)
img = cv2.resize(img,(1024,500))

cv2.imwrite("im.png",img)