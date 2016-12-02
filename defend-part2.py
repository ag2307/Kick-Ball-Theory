import cv2
import numpy as np;
import sys
from matplotlib import pyplot as plt
import math
import time
import pxssh

s = pxssh.pxssh()
if not s.login ('192.168.43.148', 'pi', 'raspberry'):
    print "SSH session failed on login."
    print str(s)
else:
    print "SSH session login successful"
    #s.sendline ('python blink.py 1 0 80')
    



#--------------------------------------------------------------------------

#Constants Decleration 														

LOWball=np.array([50,125,216])
UPball=np.array([63,141,239])

LOWbot=np.array([0,103,169])  # will actually have to do 2 of them bec of the two colours
UPbot=np.array([15,132,211])  #orange 

LOWbot1=np.array([57,83,155])
UPbot1=np.array([61,87,159])

threshArea = 50

kp=1

#---------------------------------------------------------------------

#finds the ball location

def ballp(frame):

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	b,g,r = cv2.split(frame)
	h,s,v = cv2.split(hsv)


	#Threshold the hue image to get the ball.
	ball = cv2.inRange(hsv,LOWball,UPball)

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
	centroid = (-1,-1,0)

	# Use centroid if it exists
	if centroid_x != None and centroid_y != None:

		centroid = (centroid_x,centroid_y,0)

	#cv2.circle(bgr,(centroid_x,centroid_y),10,(0,255,0),-1)

	return centroid;

#-------------------------------------------------------------------------

def blurimg():
  img = cv2.imread('image.jpg') 
  kernel = np.ones((5,5),np.float32)/25
  dst = cv2.filter2D(img,-1,kernel)
#LPF helps in removing noises, blurring the images etc. HPF filters helps in finding edges in the images.- this is a LPF here - kd
  cv2.imshow("blur",dst)
  #return dst

  #-------------------------------------------------------------------------

#finds the bot location

def findbot(frame):

	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

	#masking everything else other than colour1
	mask=cv2.inRange(hsv,LOWbot,UPbot)

	#applying gaussian blur
	bot=cv2.GaussianBlur(mask,(5,5),0) 
	bot=cv2.GaussianBlur(bot,(5,5),0) 
	bot=cv2.GaussianBlur(bot,(5,5),0) 
	# smoothing the image - http://docs.opencv.org/trunk/d4/d13/tutorial_py_filtering.html

	ret, bot = cv2.threshold(bot,-1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Take the moments to get the centroid
	moments = cv2.moments(bot)
	m00 = moments['m00']
	centroid_x, centroid_y = None, None
	if m00 != 0:
		centroid_x = int(moments['m10']/m00)
		centroid_y = int(moments['m01']/m00)
	if centroid_x != None and centroid_y != None:
		centroid_value1 = (centroid_x, centroid_y,0)
	else:
		centroid_value1 =(-1,-1,0)

	mask1=cv2.inRange(hsv,LOWbot1,UPbot1)

	#applying gaussian blur
	bot1=cv2.GaussianBlur(mask1,(5,5),0) 
	bot1=cv2.GaussianBlur(bot1,(5,5),0) 
	bot1=cv2.GaussianBlur(bot1,(5,5),0) 
	# smoothing the image - http://docs.opencv.org/trunk/d4/d13/tutorial_py_filtering.html

	ret1, bot1 = cv2.threshold(bot1,-1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Take the moments to get the centroid
	moments = cv2.moments(bot1)
	centroid_value =(0,0,0) 
	m00 = moments['m00']
	centroid_x, centroid_y = None, None
	if m00 != 0:
		centroid_x = int(moments['m10']/m00)
		centroid_y = int(moments['m01']/m00)
	if centroid_x != None and centroid_y != None:
		centroid_value2 = (centroid_x, centroid_y,0)
	else:
		centroid_value2 =(-1,-1,0)
     	

	centroid_value_x = 0.5 * (centroid_value1[0] + centroid_value2[0])
	centroid_value_y = 0.5 * (centroid_value1[1] + centroid_value2[1])
	centroid_value_z = 0.5 * (centroid_value1[2] + centroid_value2[2])

	centroid_value=(int(centroid_value_x),int(centroid_value_y),int(centroid_value_z))
	
 
	return centroid_value 



#--------------------------------------------------------------------------

#my PID control

def move(x_drop,y_drop,x_bot,y_bot):
	d=math.sqrt((x_bot-x_drop)**2+(y_drop-y_bot)**2)
	u=kp*d
	print d
	left_max=
	right_max=

	if(x_drop==-1 or x_bot==-1 or y_drop==-1 or y_bot==-1):
		u=0

    if(x_bot>left_max && x_bot<right_max):

    	if(d<15):
			u=0
		elif(d<50 and d>=15):
			u=int(math.tan(0.0523*d)*61)    
		else:
			u=150
	else:
		u=0


	if (x_drop>x_bot):
		di=1  #here di is the direction in whuch the bot should move with u velocity ; y_drop -  the drop point of the ball - kd
	elif (x_drop<x_bot):
		di=-1
	else:
		di=0

	retvalue = (u,di,0)

	return retvalue  # u is the velocity 

#---------------------------------------------------------------------------------



dropP=(-1,-1,0)
def main():
	ball=(0,0,0)


	#rospy.init_node("main")

	#ref=time.time()

	cap = cv2.VideoCapture(1)
	#time.sleep(2)

	
   # while not rospy.is_shutdown():
	
		#Get the frame
	while(1):
		ret, bgr = cap.read()
		while (ret==0):
			ret, bgr = cap.read()


			#copying data to avoid loss of information
		bgr2=np.copy(bgr)
		bgr3=np.copy(bgr)

		bot =findbot(bgr3)
		ball=ballp(bgr2)

			
		hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

		cv2.circle(bgr,(bot[0],bot[1]),10,(0,255,0),-1) #green
		
		cv2.circle(bgr,(ball[0],ball[1]),10,(0,0,255),-1) #red 
		
	
		vel_data=move(ball[0],ball[1],bot[0],bot[1])
		
		if(vel_data[1]==1):
			s.sendline('python blink.py -1 0 '+str(vel_data[2])) #
			print "FORWARD"

		elif(vel_data[1]==-1):
			s.sendline('python blink.py 1 0 ' +str(vel_data[2]))  #
			print "BACK"
			

		cv2.imshow('main',bgr)
		

		if cv2.waitKey(32) & 0xFF== 27:
			cap.release()
			cv2.destroyAllWindows()
			break

		# When everything done, release the capture
		

def dropPointLocator(point_ball,point_bot):
	global dropP
	dropP=(int(point_bot[0]),int(point_ball[1]),0)

#IF __main__ RUN THE MAIN FUNCTION. THIS IS THE MAIN THREAD
if __name__ == '__main__':
	main()
