import numpy as np
import cv2
import social_force_calculation
import bag_of_words_analysis
import lda
import matplotlib.pyplot as plt
import time


'''
This is the reproduction of "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah
The document is attached to project

validation_analysis.py includes functionality testing approach

References

1. "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah

'''

def show_data(videoUrl,resize):
    xdata = []
    ydata = []

    plt.show()

    axes = plt.gca()
    axes.set_xlim(0, 6500)
    axes.set_ylim(-4000, +0)
    axes.set_xticks([1220,4650,6445], minor=False)
    axes.set_xticks([420,520,1097,1495,1670,2170,2245,2670,2795,3270,3345,3995,4095,4520,4620,5145,5195,5695,5770,6395], minor=True)
    axes.xaxis.grid(True, which='major',color='b', linestyle='-', alpha=0.8)
    axes.xaxis.grid(True, which='minor',color='r', linestyle='-', alpha=0.2)
    line, = axes.plot(xdata, ydata, 'r-')
    cam = cv2.VideoCapture(videoUrl)
    liks=np.load('LDAresult/likelihood.npy')
    print np.amax(liks), np.amin(liks)
    j=0
    i=0
    count=0
    while (cam.isOpened()):
        ret, img = cam.read()
        if not ret:
            break
        if(i>=liks.shape[0]):
            break
        print liks[i]

        xdata.append(count)
        ydata.append(liks[i])
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0)
        count+=1

        if j==10:
            i+=1
            j=0
        img = cv2.resize(img, (0, 0), fx=resize, fy=resize)

        cv2.imshow('video', img)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
        j+=1
    plt.show()
    cam.release()


if __name__ == '__main__':
    import sys

    show_data('Crowd-Activity-All.avi', 1)