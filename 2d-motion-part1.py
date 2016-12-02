	#!/usr/bin/env python

import pxssh
import cv2
import numpy as np
import sys
#import rospy
from matplotlib import pyplot as plt
#from geometry_msgs.msg import Point
import time
from math import *

#global variables
global angle1,error,end_correction,centroid_x3,centroid_y3,cnt
angle1=0
error=15
cnt=0
end_correction=10

s = pxssh.pxssh()
if not s.login ('192.168.43.148', 'pi', 'raspberry'):
	print "SSH session failed on login."
	print str(s)
else:
	print "SSH session login successful"


# For OpenCV2 image display
WINDOW_NAME1 = 'Kick ball Theory'
#front	
lower_pink = np.array([127,111,148])		#Pink
upper_pink = np.array([171,169,247])
#back	
lower_green = np.array([61,147,190])
upper_green = np.array([127,209,255])

#lower_ball= np.array([0,180,0])  # Orange Ball-almost perfect
#upper_ball = np.array([10,255,255])    ## Values have to be changed
lower_ball= np.array([0,0,156])  # white Ball-almost perfect
upper_ball = np.array([200,60,255])    ## Values have to be changed


threshArea = 50

#----------------------------------------------------------------------------------------->
#clockwise angle finding function
def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_between(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return 360-inner
    else: # if the det > 0 then A is immediately clockwise of B
        return inner

#----------------------------------------------------------------------------------------->

#finds the ball location

def ballp(frame):

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	b,g,r = cv2.split(frame)
	h,s,v = cv2.split(hsv)


	#Threshold the hue image to get the ball.
	ball = cv2.inRange(hsv,lower_ball,upper_ball)

	#erode to reduce noise
	kernel = np.ones((5,5),np.uint8)
	erosion = cv2.erode(ball,kernel,iterations = 1)
	#followed by dilation to increase the size of final ball
	dilation = cv2.dilate(erosion,kernel,iterations = 1)


	# Take the moments to get the centroid
	moments = cv2.moments(dilation)
	m00 = moments['m00']
	centroid_x, centroid_y = None, None
	if m00 != 0:
		centroid_x = int(moments['m10']/m00)
		centroid_y = int(moments['m01']/m00)

	# Assume no centroid
	centroid = (-1,-1)

	# Use centroid if it exists
	if centroid_x != None and centroid_y != None:

		centroid = (centroid_x,centroid_y)

	#cv2.circle(bgr,(centroid_x,centroid_y),10,(0,255,0),-1)

	return centroid,centroid_x,centroid_y

#-------------------------------------------------------------------------

#finds the bot location

def findbot(frame):
	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	#masking everything else other than green
	mask=cv2.inRange(hsv,lower_pink,upper_pink)

	#applying gaussian blur
	bot=cv2.GaussianBlur(mask,(5,5),0) 
	bot=cv2.GaussianBlur(bot,(5,5),0) 
	bot=cv2.GaussianBlur(bot,(5,5),0) 
	# smoothing the image - http://docs.opencv.org/trunk/d4/d13/tutorial_py_filtering.html

	ret, bot = cv2.threshold(bot,-1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Take the moments to get the centroid
	moments = cv2.moments(bot)
	m00 = moments['m00']
	centroid_x1, centroid_y1 = None, None
	if m00 != 0:
		centroid_x1 = int(moments['m10']/m00)
		centroid_y1 = int(moments['m01']/m00)
	if centroid_x1 != None and centroid_y1 != None:
		centroid_value1 = (centroid_x1, centroid_y1)
	else:
		centroid_value1 =(-1,-1)
		centroid_x1=0
		centroid_y1=0

	mask1=cv2.inRange(hsv,lower_green,upper_green)

	#applying gaussian blur
	bot1=cv2.GaussianBlur(mask1,(5,5),0) 
	bot1=cv2.GaussianBlur(bot1,(5,5),0) 
	bot1=cv2.GaussianBlur(bot1,(5,5),0) 
	# smoothing the image - http://docs.opencv.org/trunk/d4/d13/tutorial_py_filtering.html

	ret1, bot1 = cv2.threshold(bot1,-1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Take the moments to get the centroid
	moments = cv2.moments(bot1)
	m00 = moments['m00']
	centroid_x2, centroid_y2= None, None
	if m00 != 0:
		centroid_x2 = int(moments['m10']/m00)
		centroid_y2 = int(moments['m01']/m00)
	if centroid_x2 != None and centroid_y2 != None:
		centroid_value2 = (centroid_x2, centroid_y2)
	else:
		centroid_value2 =(-1,-1)
		centroid_x2=0
		centroid_y2=0
    

	#centroid_value(0) = 0.5 * (centroid_value1(0) + centroid_value2(0))
	#centroid_value(1) = 0.5 * (centroid_value1(1) + centroid_value2(1))
	#centroid_value(2) = 0.5 * (centroid_value1(2) + centroid_value2(2))
	
 
	return centroid_value1,centroid_x1,centroid_y1,centroid_value2,centroid_x2,centroid_y2


#--------------------------------------------------------------------------

def initial(centroid_value1,centroid_value2,centroid):
	ctr4 = (centroid_value2[0]-centroid_value1[0],centroid_value2[1]-centroid_value1[1])
	ctr5 = (centroid[0]-centroid_value1[0], centroid[1]-centroid_value1[1])
	ctr6 = (1,0)
	angle= angle_between(ctr4,ctr5)
	centroid_x3=centroid[0]
	centroid_y3=centroid[1]	
#	print angle
	return angle,centroid[0],centroid[1]

#--------------------------------------------------------------------------
def move(centroid_x1,centroid_x2,centroid_y1,centroid_y2,angle,centroid_x3,centroid_y3):
	#angle=
	if centroid_x2 != None and centroid_y2 != None and centroid_y1 != None and centroid_x1 != None and centroid_y3 != None and centroid_x3 != None:
		angle1=angle_between((centroid_x2-centroid_x1,centroid_y2-centroid_y1),(centroid_x3-centroid_x1,centroid_y3-centroid_y1))
	else:
		return
	print angle
#	end_correction=10
	cnt=1;
	if centroid_x2 != None and centroid_y2 != None and centroid_y1 != None and centroid_x1 != None and centroid_y3 != None and centroid_x3 != None:
		print centroid_x1,centroid_y1
		print centroid_x2,centroid_y2
		print centroid_x3,centroid_y3
	
	print angle1

	if(angle>180 and angle<=270):	
			if(abs(centroid_x3-centroid_x1)>=55 and abs(centroid_y2-centroid_y3)>=90):
				print "forward--1"
				s.sendline('python blink.py 1 0 80')
				return
			elif(abs(centroid_x1-centroid_x2)>=4 and abs(centroid_y2-centroid_y3)>=90):
				print "right"
				s.sendline('python blink.py 0 1 0')
				return
			elif(abs(centroid_y2-centroid_y3)>=77):
				print "forward--2"
				s.sendline('python blink.py 1 0 80')
				return
			elif(abs(centroid_y2-centroid_y3)<77 and abs(centroid_y1-centroid_y2)>=5):
				print "left"
				s.sendline('python blink.py 0 -1 0')
				return
			elif(abs(centroid_y2-centroid_y3)<77 and abs(centroid_y1-centroid_y2)<=8 and abs(centroid_x3-centroid_x2)>12):
				print "forward--3"
				s.sendline('python blink.py 1 0 80')				
				return
			else:
				print "stop"
				s.sendline('python blink.py 0 0 0')
				s.sendline('python blink.py 0 0 1')
	if(angle>270 and angle<=360):	
			if(abs(centroid_x3-centroid_x1)>=55 and abs(centroid_y2-centroid_y3)>=90):
				print "backward--1"
				s.sendline('python blink.py -1 0 80')
				return
			elif(abs(centroid_x1-centroid_x2)>=4 and abs(centroid_y2-centroid_y3)>=90):
				print "left"
				s.sendline('python blink.py 0 -1 0')
				return
			elif(abs(centroid_y2-centroid_y3)>=25):
				print "backward--2"
				s.sendline('python blink.py -1 0 80')
				return
			elif(abs(centroid_y2-centroid_y3)<25 and abs(centroid_y1-centroid_y2)>=5):
				print "right"
				s.sendline('python blink.py 0 1 0')
				return
			elif(abs(centroid_y2-centroid_y3)<25 and abs(centroid_y1-centroid_y2)<=8 and abs(centroid_x3-centroid_x1)>12):
				print "backward--3"
				s.sendline('python blink.py -1 0 80')				
				return
			else:
				print "stop"
				s.sendline('python blink.py 0 0 0')
				s.sendline('python blink.py 0 0 1')

#-------------------------------------------------------------------------

def main():
	ball=(0,0,0)
	centroid_x3=0
	centroid_y3=0
	
	#rospy.init_node("main")

	#ref=time.time()

	cap = cv2.VideoCapture(1)
	#time.sleep(2)

	
   # while not rospy.is_shutdown():
#	fourcc = cv2.VideoWriter_fourcc(*'XVID')
#	out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (640,480))
		#Get the frame
	i=0;
	angle=0
	while(1):
		ret, bgr = cap.read()
		while (ret==0):
			ret, bgr = cap.read()
#  	 	if ret==True:
#      			frame = cv2.flip(bgr,0)

        # write the flipped frame
#        	out.write(frame)
		bgr2=np.copy(bgr)
		bgr3=np.copy(bgr)
		bgr4=np.copy(bgr)
		centroid_value1,centroid_x1,centroid_y1,centroid_value2,centroid_x2,centroid_y2 = findbot(bgr3)
		if(i==0):
			centroid,centroid_x,centroid_y=ballp(bgr2)
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
		if centroid_x2 != None and centroid_y2 != None and centroid_y1 != None and centroid_x1 != None:
			a=(centroid_x1+centroid_x2)/2
			b=(centroid_y1+centroid_y2)/2
		else:
			a=0
			b=0

		cv2.circle(bgr,(a,b),10,(0,255,0),-1) #green
		print a,b
		if centroid_x != None and centroid_y != None:
			a=centroid_x
			b=centroid_y
		else:
			a=0
			b=0
		cv2.circle(bgr,(a,b),10,(255,0,0),-1)   #blue
		print a,b

		
		if(i==0):	#copying data to avoid loss of information
			angle,centroid_x3,centroid_y3=initial(centroid_value1,centroid_value2,centroid)
			i=i+1
		elif(i<=500):
			move(centroid_x1,centroid_x2,centroid_y1,centroid_y2,angle,centroid_x3,centroid_y3)
			i=i+1
		else:
			cap.release()
			cv2.destroyAllWindows()
			break
		
		cv2.imshow('main',bgr)		
		if cv2.waitKey(32) & 0xFF== 27:
			cap.release()
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':
	main()
