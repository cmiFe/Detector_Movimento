import cv2
import numpy as np


video = cv2.VideoCapture(0)


mascaraBG = cv2.createBackgroundSubtractorMOG2(300, 200, True)

contaFrame = 0

while(1):
	ret, frame = video.read()
	contaFrame += 1
	novoFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
	fgmask = mascaraBG.apply(novoFrame)
	count = np.count_nonzero(fgmask)
	if (contaFrame > 1 and count > 1600):
		print('Movimento detectado')
		cv2.putText(novoFrame, 'Movimento detectado', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('Frame', novoFrame)
	cv2.imshow('Mask', fgmask)
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

video.release()
cv2.destroyAllWindows()
