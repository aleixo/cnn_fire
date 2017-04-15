import cv2
from nn_test import NnTest
from fcm_manager import sendPush
import sys
import os

cap = cv2.VideoCapture(0)

classifier = NnTest()

if not classifier.areFilesAvailable():

	print("[TEST ON CAM] Will not continue")
	sys.exit()

while(True):
    ret, frame = cap.read()
    image,preds = classifier.predictOnImage(frame)
    cv2.imshow('Classification',image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def removeNetworkFiles():

	projectDir = os.listdir("./")
	for file in projectDir:
		if file.endswith("h5") or file.endswith("json"):
			os.remove(file)
			print("[TEST ON CAM] Removed file named {}".format(file))

cap.release()
cv2.destroyAllWindows()
removeNetworkFiles()