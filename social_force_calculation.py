import numpy as np
import cv2
import video_analysis
import os
import skvideo.io

'''
This is the reproduction of "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah
The document is attached to project

social_force_calculation.py includes functionality of  calculation flow and social force and creating videos with overlayed flow and force

References

1. "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah

'''


'''
GET_EFFECTIVE_VELOCITY calculates effective velocity of pixel using bilinear interpolation from nearest neighbors

Arguments:
    flow - flow matrix with velocities(Vx, Vy) for every pixel
    
Returns:
    O - flow matrix with effective velocities
'''
def get_effective_velocity(flow):
    sp=flow.shape
    O = np.empty((sp[0], sp[1], sp[2]))
    for r in range(sp[0]):
        for c in range(sp[1]):
            Vx,Vy=bilinear_interpolation(r,c,flow)
            O[r,c,0]=Vx
            O[r,c,1]=Vy
    return O


'''
BILINEAR_INTERPOLATION is doing interpolation from nearest four points

Arguments:
    x - x coordinate of pixel
    y - y coordinage of pixel
    flow - flow matrix

Returns:
    Vx - interpolated x coordinate value of pixel
    Vy - interpolated y coordinate value of pixel
'''
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

'''
FORCE_CALCULATION calculates optical flow and social force

Arguments:
    videoUrl - url of video
    tau - relaxation parameter see Reference 1 page 3
    Pi - panic weight parameter see Reference 1 page 3
    resize - image resize factor

Returns:
    result - force flow matrix with size frame_count x video_image_size_1-2 x video_image_size_2-2x3(rgb)
    forceresult- force matrix with size  frame_count x video_image_size_1-2 x video_image_size_2-2

According to Reference 1 for calculation simplicity 25% of initial size of video is taken so because of that here we include resize parameter to resize image
to get flow we used opencv calcOpticalFlowFarneback method that calculates flow from two adjacent frames using Farneback algorithm
Desired velocity and social force are calculated using Reference 1 page 3 (4) (5)
To created colormap from force first the force value is calculated from (Fx, Fy) components and then this force normalized from range 0-255
to get full range of colors in image(normalization was done frame by frame)
Force picture sizes are smaller by 2 pixels from video image sizes (video_image_size_1-2)  this is done to avoid badly interpolated corner pixels
'''

def force_calculation(videoUrl,tau,Pi,resize):
    cam = cv2.VideoCapture(videoUrl)
    ret, prev = cam.read()
    if not ret:
        print 'Cannot read '+videoUrl
        cam.release()
        return False, np.array([]),np.array([])
    prev = cv2.resize(prev, (0, 0), fx=resize, fy=resize)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prevflow=np.zeros((prev.shape[0],prev.shape[1],2))
    result=np.empty((0,prev.shape[0]-2,prev.shape[1]-2,3))
    forceresult = np.empty((0, prev.shape[0] - 2, prev.shape[1] - 2))
    fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)  # frames per second
    while (cam.isOpened()):
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        Vef=get_effective_velocity(flow)
        Vq=(1-Pi)*Vef+Pi*flow       #desired velosity Vea
        F=tau*(Vq-flow)-(flow-prevflow)/1/fps
        F1=get_Force_from_components_and_normalize_for_image(F)
        imC = cv2.applyColorMap(F1, cv2.COLORMAP_JET)
        result=np.append(result,np.array([imC]),axis=0)
        forceresult = np.append(forceresult, np.array([F1]), axis=0)
        prevflow=flow
        prevgray = gray
    cam.release()
    return True, result,forceresult

'''
GET_FORCE_FROM_COMPONENTS_AND_NORMALIZE_FOR_IMAGE is calculating force from (Fx, Fy) components and normalizing in range 0-255

Arguments:
    Force-force matrix

Returns:
    force matrix
'''
def get_Force_from_components_and_normalize_for_image(Force):
    F=np.sqrt(Force[:,:,0]**2+Force[:,:,1]**2)
    F=F[1:F.shape[0]-1,1:F.shape[1]-1]  #removing corner pixels  these have not good bilinear interpolation velocities
    F*=255.0/np.max(F)
    return np.round(F).astype(np.uint8)


'''
GET_FORCE_FLOW is calculating force from (Fx, Fy) components, normalizing in range 0-255 and opencv colormaping

Arguments:
    Force-force matrix

Returns:
    imC-opencv colormaped image matrix
'''
def get_Force_Flow(F):
    F1 = get_Force_from_components_and_normalize_for_image(F)
    imC = cv2.applyColorMap(F1, cv2.COLORMAP_JET)
    return imC

'''
GET_VIDEO_AND_CREATE_VIDEO_WITH_FORCE_AND_FLOW creates videos with flows and overlayed forces
for every video 2 video is outputing one with optical flow(yellow) and force (red) lines and
the second is colormaped force flow overlayed to actual video

Arguments:
    directory - videos directory
    tau - relaxation parameter see Reference 1 page 3
    Pi - panic weight parameter see Reference 1 page 3
    resize - image resize factor

Returns:
    creates videos in the same directory
'''
def get_video_and_create_video_with_force_and_flow(directory,tau,Pi,resize):
    videos = os.listdir(directory)
    for file in videos:
        print file
        fn=file.split(".")[0]
        fn_ext = file.split(".")[-1]
        cam = cv2.VideoCapture(directory + '/' + file)
        ret, prev = cam.read()
        if not ret:
            print 'Cannot read '+file+' continuing to next'
            cam.release()
            continue
        prev = cv2.resize(prev, (0, 0), fx=resize, fy=resize)
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prevflow = np.zeros((prev.shape[0], prev.shape[1], 2))
        fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)
        fourcc = cv2.cv.CV_FOURCC('I', 'Y', 'U', 'V')
        writer_flow = cv2.VideoWriter(directory+' Flow/'+fn + '_flow.avi', fourcc, fps, (prev.shape[1], prev.shape[0]))
        writer_overlay = cv2.VideoWriter(directory+' OverLay/'+fn + '_overlay.avi', fourcc, fps, (prev.shape[1]-2, prev.shape[0]-2))
        while (cam.isOpened()):
            ret, img = cam.read()
            if not ret:
                break
            img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            Vef = get_effective_velocity(flow)
            Vq = (1 - Pi) * Vef + Pi * flow
            F = tau * (Vq - flow) - (flow - prevflow) / 1 / fps
            imC = get_Force_Flow(F)
            writer_flow.write(video_analysis.draw_flow_with_force(gray, flow, F))
            writer_overlay.write(video_analysis.overlay_image(gray[1:gray.shape[0]-1, 1:gray.shape[1]-1], imC))
            prevflow = flow
            prevgray = gray
        cam.release()
        writer_flow.release()
        writer_overlay.release()

'''
GET_VIDEO_AND_CREATE_COLORMAP_VIDEO creates videos force color maps and for
calculation simplicity the size of image is taken of 25% of initial image size Reference 1 page 5

Arguments:
    directory - videos directory
    tau - relaxation parameter see Reference 1 page 3
    Pi - panic weight parameter see Reference 1 page 3
    resize=0.25 - image resize factor 25 % of initial image size

Returns:
    creates videos in the same directory
'''

def get_video_and_create_colormap_video(directory,tau,Pi,resize=0.25):
    videos = os.listdir(directory)
    for file in videos:
        print file
        fn=file.split(".")[0]
        fn_ext=file.split(".")[-1]
        cam = cv2.VideoCapture(directory+'/'+file)
        ret, prev = cam.read()
        if not ret:
            cam.release()
            print 'Cant read '+file+' continuing to next'
            continue
        prev = cv2.resize(prev, (0, 0), fx=resize, fy=resize)
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        prevflow = np.zeros((prev.shape[0], prev.shape[1], 2))
        fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)
        fourcc=cv2.cv.CV_FOURCC('I','Y','U','V')
        writer_flow = cv2.VideoWriter(directory+' Colormap/'+fn + '_colormap.avi', fourcc, fps, (prev.shape[1]-2, prev.shape[0]-2,))
        while (cam.isOpened()):
            ret, img = cam.read()
            if not ret:
                break
            img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            Vef = get_effective_velocity(flow)
            Vq = (1 - Pi) * Vef + Pi * flow
            F = tau * (Vq - flow) - (flow - prevflow) / 1 / fps
            imC = get_Force_Flow(F)
            writer_flow.write(imC)
            prevflow = flow
            prevgray = gray
        cam.release()
        writer_flow.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys

    #make it true if you want to create videos
    createVideos=False
    createOnlyTestVideo=True
    if createVideos:
        #normal crowd videos
        get_video_and_create_video_with_force_and_flow('Normal Crowds', 0.5, 0, 1)
        get_video_and_create_colormap_video('Normal Crowds', 0.5, 0)

        #abnormal crowd videos
        get_video_and_create_video_with_force_and_flow('Abnormal Crowds', 0.5, 0, 1)
        get_video_and_create_colormap_video('Abnormal Crowds', 0.5, 0)

        #crowd dataset extra
        get_video_and_create_video_with_force_and_flow('Crowd Dataset - extra', 0.5, 0, 1)
        get_video_and_create_colormap_video('Crowd Dataset - extra', 0.5, 0)

    if createOnlyTestVideo:
        # test dataset
        get_video_and_create_video_with_force_and_flow('Test Dataset Crowd', 0.5, 0, 1)
        get_video_and_create_colormap_video('Test Dataset Crowd', 0.5, 0)



