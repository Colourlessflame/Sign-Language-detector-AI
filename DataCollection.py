#This file is used to collect the data. You can classify it into A B C or something like that.
#Libraries
import os
import cv2

#I'm writing it as an Absolute file path since I don't trust my PC. If you want you can just change it to Sign Language detector/trainingdata
DATA_DIR = 'C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/trainingdata'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3 #Change this for more data samples and more letter
FRAME_RATE=25
dataset_size = 100

camera = cv2.VideoCapture(0)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print(f'Collecting data for class {j}')

    
    while True:
        read, frame = camera.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('Window', frame)

        #Change the small q to any character if you want it to stop getting that data
        if cv2.waitKey(25) & 0xFF== ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = camera.read()
        cv2.imshow('Window', frame)
        cv2.waitKey(FRAME_RATE)
        #This saves the data to a file called training data with 100 images for each alphabet.
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

camera.release()
cv2.destroyAllWindows()