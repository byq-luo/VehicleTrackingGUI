import numpy as np
import cv2
#from sort.py import rama
import time

cap = cv2.VideoCapture('Night.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))

while(cap.isOpened()):
	
	ret, frame = cap.read()
	
	frame = cv2.line(frame, (100,360), (1200,360), (0,0,255), 6)
	
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	frame = cv2.putText(frame, ('CAR'), (10,500), font, 4, (255,255,255),2,cv2.LINE_AA)
	out.write(frame)
	cv2.imshow('frame', frame)
	
	if cv2.waitKey(25) == ord('q'):
		break
