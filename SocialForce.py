import numpy as np
import cv2
import time
import Video

def get_Desired_Velocity(flow):
    sp=flow.shape
    O = np.empty((sp[0], sp[1], sp[2]))
    for r in range(sp[0]):
        for c in range(sp[1]):
            Vx,Vy=bilinear_interpolation(r,c,flow)
            O[r,c,0]=Vx
            O[r,c,1]=Vy
    return O

def bilinear_interpolation(x, y, flow):
    #Interpolate (x,y) from values associated with four points.
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation
    im=x if x-1<0 else x-1
    ip=x if x+1>=flow.shape[0] else x+1
    jm=y if y-1<0 else y-1
    jp=y if y+1>=flow.shape[1] else y+1

    point1x=flow[im,jm,0]
    point1y=flow[im,jm,1]

    point2x = flow[im, jp, 0]
    point2y = flow[im, jp, 1]

    point3x = flow[ip, jm, 0]
    point3y = flow[ip, jm, 1]

    point4x = flow[ip, jp, 0]
    point4y = flow[ip, jp, 1]

    Vx=(point1x*(x-im)*(y-jm)+point2x*(x-im)*(jp-y)+point3x*(ip-x)*(y-jm)+point4x*(ip-x)*(jp-y))/((ip-im)*(jp-jm))
    Vy = (point1y * (x - im) * (y - jm) + point2y * (x - im) * (jp - y) + point3y * (ip - x) * (y - jm) + point4y * (
    ip - x) * (jp - y)) / ((ip - im) * (jp - jm))
    return Vx, Vy

if __name__ == '__main__':
    import sys

    cam = cv2.VideoCapture("Normal Crowds/879-38_l.mov")
    ret, prev = cam.read()
    prev = cv2.resize(prev, (0, 0), fx=2, fy=2)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevflow=np.empty((prev.shape[0],prev.shape[1],2))
    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (0, 0), fx=2, fy=2)
        vis = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        Vef=get_Desired_Velocity(flow)
        Vq=(1-0.5)*Vef+0.5*flow
        tau=20
        F=tau*(Vq-flow)-(flow-prevflow)/1/24   # 24 frames per second
        print flow
        prevflow=flow
        prevgray = gray
        cv2.imshow('flow', Video.draw_flow_with_force(gray, flow,F))

        #cv2.imshow('Image', vis)
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break


    cv2.destroyAllWindows()