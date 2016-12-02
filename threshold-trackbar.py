import numpy
import cv2

def nothing(x):
	pass

cap=cv2.VideoCapture(1)
cv2.namedWindow("Threshold")
cv2.createTrackbar("H-Low","Threshold",0,255,nothing)
cv2.createTrackbar("H-Up","Threshold",0,255,nothing)
cv2.createTrackbar("S-Low","Threshold",0,255,nothing)
cv2.createTrackbar("S-Up","Threshold",0,255,nothing)
cv2.createTrackbar("V-Low","Threshold",0,255,nothing)
cv2.createTrackbar("V-Up","Threshold",0,255,nothing)
while(1):
	ret,frame=cap.read()

	hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	hl=cv2.getTrackbarPos("H-low","Threshold")
	hu=cv2.getTrackbarPos("H-Up","Threshold")
	sl=cv2.getTrackbarPos("S-Low",'Threshold')
	su=cv2.getTrackbarPos("S-Up",'Threshold')
	vl=cv2.getTrackbarPos("V-Low","Threshold")
	vu=cv2.getTrackbarPos("V-Up","Threshold")
	lower_bound=numpy.array([hl,sl,vl])
	upper_bound=numpy.array([hu,su,vu])

	mask=cv2.inRange(hsv,lower_bound,upper_bound)
	
	res=cv2.bitwise_and(frame,frame,mask=mask)

	cv2.imshow("Video",frame)
	#cv2.imshow("mask",mask)

	cv2.imshow("Threshold",res)
	k = cv2.waitKey(5)&0xFF
	if k==27:
		break
cap.release()
cv2.destroyAllWindows()
