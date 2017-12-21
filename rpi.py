# import the necessary packages
import numpy as np
# import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import pdb #python debugger
import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import json


def write_q_mat(q_matrix):
    with open("/home/pi/dc_q_learning/q_matrix.json","w") as json_write:
        json.dump(q_matrix.tolist(), json_write, indent=4)

def take_action(kl):
    action = np.argmax(r_matrix[kl,:])
    return ac
    
# Create a memory stream so photos doesn't need to be saved in a file
#stream = io.BytesIO()

global r_matrix
r_matrix = np.matrix([[0,0,0,0,0,0,0,0,0,0,0,10,20,30,40,50,60,70,80,90,100],
                      [0,0,0,0,0,0,0,0,0,10,20,30,40,50,60,70,80,90,100,90,80],
                      [0,0,0,0,0,0,0,10,20,30,40,50,60,70,80,90,100,90,80,70,60],
                      [0,0,0,0,0,10,20,30,40,50,60,70,80,90,100,90,80,70,60,50,40],
                      [0,0,0,10,20,30,40,50,60,70,80,90,100,90,80,70,60,50,40,30,20],
                      [0,10,20,30,40,50,60,70,80,900,1000,900,80,70,60,50,40,30,20,10,0],
                      [20,30,40,50,60,70,80,90,100,90,80,70,60,50,40,30,20,10,0,0,0],
                      [40,50,60,70,80,90,100,90,80,70,60,50,40,30,20,10,0,0,0,0,0],
                      [60,70,80,90,100,90,80,70,60,50,40,30,20,10,0,0,0,0,0,0,0],
                      [80,90,100,90,80,70,60,50,40,30,20,10,0,0,0,0,0,0,0,0,0],
                      [100,90,80,70,60,50,40,30,20,10,0,0,0,0,0,0,0,0,0,0,0]])
global q_matrix
with open("/home/pi/dc_q_learning/q_matrix.json") as json_read:
    q_matrix = np.array(json.load(json_read))

global frcnt
frcnt = 0

#learning rate alpha 
global alpha  
alpha = 0.4

# initialize the camera and grab a reference to the raw camera capture

camera = PiCamera()
camera.resolution = (80, 60)
camera.framerate = 10
#camera.capture(stream, format = 'jpeg')
rawCapture = PiRGBArray(camera, size=(80,60))
#buff = np.fromstring(stream.getvalue(), dtype = np.uint8)
#pdb.set_trace()
time.sleep(1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    img = frame.array
    #cv2.imshow('ander',img)        
    #edgesori = cv2.Canny(img, 90, 255)
    ##Night##blurred = cv2.pyrMeanShiftFiltering(img, 10, 41)
    ##Night##edgesblur = cv2.Canny(blurred, 90, 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edgesgray = cv2.Canny(blurred, 90, 255)
     
    #print(edgesgray)

    _, edgesthresh = cv2.threshold(gray,200 , 255, cv2.THRESH_BINARY)
    ####edgesthresh = cv2.Canny(thresh, 90, 255, apertureSize=5)
    print(edgesthresh.shape)
    ####cv2.imshow('gray',thresh)   
    ####cv2.imshow('con',edgesthresh)    
    ####_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ####cnt = contours[-1]
    # M = cv2.moments(cnt)
    # print( M )

    # cv2.imshow('edgesori',edgesori)
    # cv2.imshow('edgesblur',edgesblur)
    # cv2.imshow('edgesgray',edgesgray)
    global width
    height, width = edgesthresh.shape
    '''
    print(height)
    print(width)
    print(np.min(edgesthresh))
    print('hilleeomdacd')
    '''
    global aj
    global ak

    aj = 0
    ak = 80
    global camvar
    camvar = 1
    lft_x = []
    lft_y = []
    rit_x = []
    rit_y = []
    ####cv2.imshow('edghjj',edgesthresh)
    for i in range(height - 1, int(height / 2), -1):

        for j in range(int(width / 2), 0, -1):
            if edgesthresh[i, j] == 255:
                ####if i in lft_y:
                    ####break
                ####else:
                lft_y.append(i)
                lft_x.append(j)
                    # np.concatenate(lft_y, i)
                    # np.concatenate((lft_x, np.array([j])))
                aj = j
        '''
                    j = j - 4
                    Night
                for n in range(j, 0, -1):
                    if edgesthresh[i, n] == 255:
                        edgesthresh[i, n] = 0
        '''
        '''
        if camvar == 1:
             edgesthresh[i, int(((aj + ak) + 80 - aj) / 2)] = 200
             edgesthresh[i, int((aj + ak) / 2)] = 100
        '''
        for k in range(int(width / 2), 80):
            j = aj
            if edgesthresh[i, k] == 255:
                ####if i in rit_y:
                    ####break
                ####else:
                rit_y.append(i)
                rit_x.append(k)
                    # rit_y.np.append([i])
                    # rit_x.np.append([k])
                ak = k
            '''
                    k = k + 4
                
                camvar = 0
                for m in range(k, 160):
                    if edgesthresh[i, m] == 255:
                        edgesthresh[i, m] = 0
            '''
            if j < 0:
                j = 0
            width = (aj + ak)
            '''
                print('i=', i)
                print(int(width / 2))
                divi = ((ak - aj) / 7)
                edgesthresh[i, int(width / 2)] = 255
                for g in range(1, 7):
                    edgesthresh[i, int(aj + (g * divi))] = 125
                    if (int(width / 2) > (int(aj + ((g - 1) * divi)))):
                        if (int(width / 2) < (int(aj + (g * divi)))):
                            print(' in stage ', g)
            '''
    #cv2.waitKey(0)

    # cv2.line(edgesthresh, (0,65),(160,65),(255),1)
    # cv2.line(edgesthresh, (0,60),(160,60),(200),1)

    # cv2.line(edgethresh,
    # print(lft_x)
    # cv2.imshow('gray',gray)
    ##Night##cv2.drawContours(thresh, [cnt], -1, (0, 255, 0), 10)
    ####cv2.imshow('lo',img)
    
    ####key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
 
    # if the `q` key was pressed, break from the loop
    ####if key == ord("q"):
        ####break
    #cv2.destroyAllWindows()

    lx = np.array(lft_x)
    ly = np.array(lft_y)
    #print(ly)
    l = np.polyfit(ly, lx, 2)
    #print(l)
    lp = np.poly1d(l)
    #print(lp)

    rx = np.array(rit_x)
    ry = np.array(rit_y)
    r = np.polyfit(ry, rx, 2)
    rp = np.poly1d(r)

    #pdb.set_trace()
    '''
    global lxmin
    if np.amax(ly,axis=0)<120:
        lxmin = np.amin(lx, axis=0)
        while ((lp(lxmin)) < 120):
            print('left' ,lxmin)
            lxmin = lxmin - 1
    else:
        lxmin = np.amin(lx,axis=0)
    global rxmax
    if np.amax(ry,axis=0)<120:
        rxmax = np.amax(rx, axis=0)
        while rp(rxmax) < 120:
            print(rxmax)l
            rxmax = rxmax + 1
    else:
        rxmax = np.amax(rx, axis=0)
    '''
    datal = np.linspace(np.amin(ly, axis=0),30, 60)
    datar = np.linspace(np.amin(ry, axis=0),30, 60)
    #plt.plot(lx, ly, 'r-', rx, ry, 'b-')
    #plt.plot(np.polyval(l, datal),datal, 'b--')
    #plt.plot(np.polyval(r,datar),datar,'r--')
    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()
    global frcnt
    if frcnt == 0:
        for i in range (60,30,-5):
            re = (rp(i) - lp(i))/11
            for kl in range (0,11):
                if 40 > (lp(i) + (kl*re)):
                    
                    if 40<= (lp(i) + ((kl+1)*re)):
                        print('I am in stage ',(kl+1))
                        global st1,st2,st3,st4, act1,act2,act3,act4
                        if frcnt == 0:
                            st1 = int(kl+1)
                            act1  = np.argmax(r_matrix[kl,:])
                        elif frcnt == 1:
                            st2 = int(kl+1)
                            act2  = np.argmax(r_matrix[kl,:])
                        elif frcnt == 2:
                            st3 = int(kl+1)
                            act3 = np.argmax(r_matrix[kl, :])
                        elif frcnt == 3:
                            st4 = int(kl + 1)
                            act4 = np.argmax(r_matrix[kl, :])
                        else:
                            st5 = int(kl + 1)
                            act5 = np.argmax(r_matrix[kl, :])

                        frcnt = frcnt + 1
                        if frcnt == 5:
                            frcnt = 0
                            
                            q_matrix[st1-1,act1] = ((1 - alpha)*q_matrix[(st1-1),act1]) + alpha*(r_matrix[(st1 - 1),act1] + (0.333 * q_matrix[st2 - 1, act2]) + (0.333 * q_matrix[st3 - 1, act3]) +(0.333 * q_matrix[st4 - 1, act4]))

                            with open("/home/pi/dc_q_learning/q_matrix.json","w") as json_write:
                                json.dump(q_matrix.tolist(), json_write, indent=4)
                            
                            ####write_q_mat(q_matrix)
    else:
        st1 = 0
        st2 = 0
        st3 = 0
        st4 = 0
        st5 = 0
        st6 = 0

        act1 = 0
        act2 = 0
        act3 = 0
        act4 = 0
        act5 = 0
        act6 = 0

        exit()
    
