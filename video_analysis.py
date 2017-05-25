import numpy as np
import cv2

'''
This is the reproduction of "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah
The document is attached to project

video_analysis.py includes functionality of calculation and drawing the social force and flow on video 

References

1. "Abnormal Crowd Behavior Detection using Social Force Model" by Ramin Mehran, Alexis Oyama, Mubarak Shah

'''

'''
WRITE_TEXT_ON_IMAGE added text to image that then will be formed to video this is for add "Normal" or "Abnormal" to image

Arguments:
    img - opencv image matrix
    text - required text
'''
def write_text_on_image(img,text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (40, 40), font, 0.1, (255, 255, 255), 2)



'''
DRAW_FLOW this methods draws flow on image

Arguments:
    img - opencv image matrix
    flow - flow matrix with velocities(Vx, Vy) for every pixel
    step - step size of points with flow lines for visualization

Returns:
    vis - image with flow
'''
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    scale=1.2
    fx, fy = flow[y, x].T*scale
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 255),1)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 0, 255), -1)
    return vis

'''
DRAW_FLOW_WITH_FORCE this methods draws flow and force on image

Arguments:
    img - opencv image matrix
    flow - flow matrix with velocities(Vx, Vy) for every pixel
    force - force matrix with Forcex(Fx, Fy) for every pixel
    step - step size of points with flow and force  lines for visualization

Returns:
    vis - image with flow and force
'''

def draw_flow_with_force(img, flow, force, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    scale=3
    fx, fy = flow[y, x].T*scale
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    scale1=30
    fx1, fy1 = force[y, x].T*scale1

    lines1 = np.vstack([x, y, x + fx1, y + fy1]).T.reshape(-1, 2, 2)
    lines1 = np.int32(lines1 + 0.5)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, 0, (0, 255, 255),1)
    cv2.polylines(vis, lines1, 0, (0, 0, 255),1)

    return vis

'''
overlay_image this methods overlays one image on another using alpha channel

Arguments:
    source - image matrix
    overlay - image matrix that needs to be overlayed
Returns:
    result - overlayed image matrix
'''
def overlay_image(source, overlay):
    h, w, depth = overlay.shape
    result = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            color1 = source[i, j]
            color2 = overlay[i, j]
            alpha = 0.5
            new_color = [(1 - alpha) * color1 + alpha * color2[0],
                         (1 - alpha) * color1 + alpha * color2[1],
                         (1 - alpha) * color1 + alpha * color2[2]]
            result[i, j] = new_color
    return result


if __name__ == '__main__':
    import sys

# here and example of video with optical flow
    resize=2
    cam = cv2.VideoCapture("Normal Crowds/879-38_l.mov")
    ret, prev = cam.read()
    if not ret:
        print 'Cant read file'
    prev = cv2.resize(prev, (0, 0), fx=resize, fy=resize)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while True:
        ret, img = cam.read()
        img = cv2.resize(img, (0, 0), fx=resize, fy=resize)
        vis = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray, flow))
        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break

    cv2.destroyAllWindows()