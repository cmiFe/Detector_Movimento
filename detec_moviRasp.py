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
rawCapture = PiRGBArray(camera, size=(640, 480))
# Keeps track of what frame we're on
frameCount = 0

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text
            imagem_frame = frame.array
            imagem_frame = imagem_frame[100:100+250,120:120+250]
            gray = cv2.cvtColor(imagem_frame, cv2.COLOR_BGR2GRAY)

            frameCount += 1
            # Resize the frame
            resizedFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

            # Get the foreground mask
            fgmask = fgbg.apply(resizedFrame)

            # Count all the non zero pixels within the mask
            count = np.count_nonzero(fgmask)

            print('Frame: %d, Pixel Count: %d' % (frameCount, count))

            if (frameCount > 1 and count > 5000):
                print('Movimento detectado')
                cv2.putText(resizedFrame, 'Movimento detectado', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Frame', resizedFrame)
            cv2.imshow('Mask', fgmask)
            key = cv2.waitKey(1) & 0xFF
            rawCapture.truncate(0)

            if key == ord("q"):
                break
