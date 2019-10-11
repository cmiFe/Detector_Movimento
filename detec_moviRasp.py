import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
# Video Capture
# capture = cv2.VideoCapture(0)

# History, Threshold, DetectShadows
# fgbg = cv2.createBackgroundSubtractorMOG2(50, 200, True)
fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
video = PiRGBArray(camera, size=(640, 480))
# Keeps track of what frame we're on
contaFrame = 0

for frame in camera.capture_continuous(video, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            imagem_frame = frame.array
            imagem_frame = imagem_frame[100:100+250,120:120+250]
            gray = cv2.cvtColor(imagem_frame, cv2.COLOR_BGR2GRAY)

            contaFrame += 1
            # Resize the frame
            novoFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

            # Get the foreground mask
            fgmask = fgbg.apply(novoFrame)

            # Count all the non zero pixels within the mask
            contador = np.count_nonzero(fgmask)

            if (contaFrame > 1 and contador > 5000):
                print('Movimento detectado')
                cv2.putText(novoFrame, 'Movimento detectado', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Frame', novoFrame)
            cv2.imshow('Mask', fgmask)
            key = cv2.waitKey(1) & 0xFF
            video.truncate(0)

            if key == ord("q"):
                break
