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
global angle1,angle,error,end_correction
angle=0
angle1=0
error=10
end_correction=10
'''
s = pxssh.pxssh()
if not s.login ('192.168.43.148', 'pi', 'raspberry'):
	print "SSH session failed on login."
	print str(s)
else:
	print "SSH session login successful"
'''
# For OpenCV2 image display
WINDOW_NAME1 = 'Kick ball Theory'
#front	
lower_pink = np.array([145,72,200])		#Pink
upper_pink = np.array([201,136,255])
#back	
lower_green = np.array([76,111,161])
upper_green = np.array([107,194,255])

lower_ball= np.array([0,81,0])  # Orange Ball-almost perfect
upper_ball = np.array([16,255,255])    ## Values have to be changed
#lower_ball= np.array([0,0,162])  # white Ball-almost perfect
#upper_ball = np.array([203,65,255])    ## Values have to be changed

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
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

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

	print "hello"
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
'''
def initial(centroid_value1,centroid_value2,centroid):
	ctr4 = (centroid_value2[0]-centroid_value1[0],centroid_value2[1]-centroid_value1[1])
	ctr5 = (centroid[0]-centroid_value1[0], centroid[1]-centroid_value1[1])
	ctr6 = (1,0)
	angle= angle_between(ctr4,ctr5)
	angle1=angle_between(ctr6,ctr4)

	while(abs(centroid_y1-centroid_y2)<error):
		if(angle1<=90):
			#point1=(0,-1,0)
			s.sendline('python blink.py 0 -1 0')
			print "yo yo yo yo yo"
			print "left\n"
		elif(angle1<=180):
			#point1=(0,-1,0)
			s.sendline('python blink.py 0 -1 0')
			print "yo yo yo yo yo"
			print "left\n"
		elif(angle1<=270):
			#point1=(0,1,0)
			s.sendline('python blink.py 0 1 0')
			print "yo yo yo yo yo"
			print "right\n"  
		elif(angle1<=360):
			#point1=(0,1,0)
			s.sendline('python blink.py 0 1 0')
			print "yo yo yo yo yo"
			print "right"
	return
'''
'''
#--------------------------------------------------------------------------
def move(centroid_x1,centroid_x2,centroid_y1,centroid_y2,centroid_x3,centroid_y3):
	print angle1
	end_correction=5

	while(centroid_y3>=centroid_y2): 
		if(angle<=90):
			if(abs(centroid_x2-centroid_x1)<error):
				point1=(0,1,0)
				print "right"
				s.sendline('python blink.py 0 1 0')
			elif(abs(centroid_y1-centroid_y2)<error):
				point(0,-1,0)
				print "left"
				s.sendline('python blink.py 0 -1 0')		
			elif(centroid_y2<centroid_y3+end_correction):
				point1=(1,0,vel)
				print "back"
				s.sendline('python blink.py -1 0 60')
			elif(centroid_x2<centroid_x3+end_correction):
				point1=(1,0,vel)
				print "back"
				s.sendline('python blink.py -1 0 60')
			else:
				point1=(0,0,0)
				print "stop"
				s.sendline('python blink.py 0 0 0')
		elif(angle<=180):
		
			if(abs(centroid_x2-centroid_x1)<error):
				point1=(0,1,0)
				print "left"
				s.sendline('python blink.py 0 -1 0')
			
			elif(abs(centroid_y1-centroid_y2)<error):
				point(0,1,0)
				print "right"
				s.sendline('python blink.py 0 1 0')		
			
			elif(centroid_y2<centroid_y3+end_correction):
				point1=(1,0,vel)
				print "forward"
				s.sendline('python blink.py 1 0 60')

			elif(centroid_x2>centroid_x3-end_correction):
				point1=(1,0,vel)
				print "forward"
				s.sendline('python blink.py 1 0 60')
			
			else:
				point1=(0,0,0)  
				print "stop"   
				s.sendline('python blink.py 0 0 0')
		elif(angle<=270):
			if(abs(centroid_x2-centroid_x1)<error):
				point1=(0,-1,0)
				print "right"
				s.sendline('python blink.py 0 1 0')
		
			elif(abs(centroid_y1-centroid_y2)<error):
				point(0,-1,0)
				print "left"
				s.sendline('python blink.py 0 -1 0')
			elif(centroid_y2>centroid_y3+end_correction):
				point1=(1,0,vel)
				print "forward"
				s.sendline('python blink.py 1 0 60')

			
			elif(centroid_x2>centroid_x3-end_correction):
				point1=(1,0,vel)
				print "forward"
				s.sendline('python blink.py 1 0 60')
			
			else:
				point1=(0,0,0)
				print "stop"
				s.sendline('python blink.py 0 0 0')
		elif(angle<=360):
			if(abs(centroid_x2-centroid_x1)<error):
				point1=(0,-1,0)
				print "left"
				s.sendline('python blink.py 0 -1 0')
			
			elif(abs(centroid_y1-centroid_y2)<error):
				point(0,1,0)
				print "right"
				s.sendline('python blink.py 0 1 0')
			elif(centroid_y2>centroid_y3+end_correction):
				point1=(1,0,vel)
				print "back"
				s.sendline('python blink.py -1 0 60')
		
			elif(centroid_x2<centroid_x3+end_correction):
				point1=(1,0,vel)
				print "back"
				s.sendline('python blink.py -1 0 60')
				
			else:
				point1=(0,0,0)
				print "stop"
				s.sendline('python blink.py 0 0 0')

'''
#--------------------------------------------------------------------------

def main():
	ball=(0,0,0)


	#rospy.init_node("main")

	#ref=time.time()

	cap = cv2.VideoCapture(1)
	print "first"
	#time.sleep(2)

	
   # while not rospy.is_shutdown():
	
		#Get the frame
	i=0;
	while(1):
		ret, bgr = cap.read()
		while (ret==0):
			ret, bgr = cap.read()
		bgr2=np.copy(bgr)
		bgr3=np.copy(bgr)
		bgr4=np.copy(bgr)
		centroid_value1,centroid_x1,centroid_y1,centroid_value2,centroid_x2,centroid_y2 = findbot(bgr3)
		centroid,centroid_x,centroid_y=ballp(bgr2)

			#gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)



		#table=cv2.inRange(hsv,LOWtable,UPtable)
		#table=cv2.GaussianBlur(table,(15,15),0)
		#table=cv2.GaussianBlur(table,(15,15),0)
		#table=cv2.GaussianBlur(table,(15,15),0)
			# ret, table = cv2.threshold(table,100,255,cv2.THRESH_BINARY)
		#ret, table = cv2.threshold(table,-1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

			# Find the coordinates of the drop position

		#dropPointLocator(ball,bot)
		#print "hello"

		#cv2.circle(bgr,(int(dropP[0]),int(dropP[1])),10,(255,0,0),-
		if centroid_x2 != None and centroid_y2 != None and centroid_y1 != None and centroid_x1 != None:
			a=(centroid_x1+centroid_x2)/2
			b=(centroid_y1+centroid_y2)/2
		else:
			a=0
			b=0

		cv2.circle(bgr,(a,b),10,(0,255,0),-1) #green
		#cv2.circle(bgr,(centroid_x,centroid_y),10,(0,0,255),-1) #red 
		print a,b
		if centroid_x != None and centroid_y != None:
			a=centroid_x
			b=centroid_y
		else:
			a=0
			b=0
		cv2.circle(bgr,(a,b),10,(255,0,0),-1)   #blue
		print a,b

		'''
		if(i==0):	#copying data to avoid loss of information
			initial(centroid_value1,centroid_value2,centroid)
			print "hey hi"
			i=i+1
		else:
			move(centroid_x1,centroid_x2,centroid_y1,centroid_y2,centroid_x,centroid_y)
			print "hey hi 2"

			#getting motor velocity and publishing it
		'''
		'''
		vel_data=move(ball[0],ball[1],bot[0],bot[1])

		
		if(vel_data[1]==1):
			#s.sendline('python blink.py 1 0 '+str(vel_data[0]))
			print "forward"
			print vel_data[0]

		elif(vel_data[1]==-1):
			#s.sendline('python blink.py -1 0 '+str(vel_data[0]))
			print "back"
			print vel_data[0]
		'''	
			#pub.publish(vel_data) ######THIS IS THE FINAL DATA TO BE SENT
		cv2.imshow('main',bgr)		
		if cv2.waitKey(32) & 0xFF== 27:
			cap.release()
			cv2.destroyAllWindows()
			break

if __name__ == '__main__':
	main()
