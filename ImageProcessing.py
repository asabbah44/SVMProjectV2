import cv2
import matplotlib.pylab as plt

image = cv2.imread('D:/data/train/cat/0090.jpg')
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
for i, col in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([hsvImage], [i], None, [32], [0,32])
    plt.plot(hist, color=col)
    plt.xlim([0, 32])

plt.show()
print(hist.shape)