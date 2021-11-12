import cv2
import numpy as np
from WindowString import Window
from djitellopy import tello

cap = cv2.VideoCapture(0)

text = Window('uu',500,500)

params = {}
max_i = [0,0,0]
min_i = [0,0,0]
saveA = max_i
saveI = min_i
def draw(event,x,y,flags,para):
    global max_i,min_i,saveA,saveI
    img = para['img'][y][x]
    txt = "(R:"+str(img[2])+" G:"+str(img[1])+" B:"+str(img[0])+")"
    text.write('a',txt,0,0)
    img = cv2.cvtColor(np.uint8([[img]]),cv2.COLOR_RGB2HSV)[0][0]
    
    txt = "(H:"+str(img[0])+" S:"+str(img[1])+" V:"+str(img[2])+")"
    text.write('b',txt,0,1)
    text.write('c',str(flags),0,2)

    if flags == 1:
        h,s,v = img
        max_i = [max(max_i[0],h),max(max_i[1],s),max(max_i[2],v)]
        min_i = [min(min_i[0],h),min(min_i[1],s),min(min_i[2],v)]
     #   min_i = min(min_i,img)
        text.write('max','max:'+str(max_i),0,4)
        text.write('min','min:'+str(min_i),0,3)
        saveA = max_i
        saveI = min_i
    else:
        max_i = img
        min_i = img
        



me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',draw,params)

while(1):
    time22 = cv2.getTickCount()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    # Take each frame
   
    #_, frame = cap.read()
    frame = img
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # define range of blue color in HSV
#    lower_blue = np.array([0,0,50])
#    upper_blue = np.array([179,10,255])
#    lower_blue = np.array([110,50,50])
#    upper_blue = np.array([130,255,255])
    lower_blue = np.array([35,114,20])
    upper_blue = np.array([51,255,209])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    params['img'] = frame
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    
    text.show()
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k == ord('m'):
        print(str(saveI))
        print(str(saveA))
    
    time33 = cv2.getTickCount()
    fps = round( cv2.getTickFrequency()/(time33 - time22) )
    text.write('time','FPS:'+str(fps),0,7)


cv2.destroyAllWindows()