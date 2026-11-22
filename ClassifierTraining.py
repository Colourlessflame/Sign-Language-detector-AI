#Usedd for training the classifier for detecting hang movement
print("Importing libraries")

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy

print("Done importing")



#absolute file path used as I'm not risking it for my laptop, you can change if you need to
data_Dict=pickle.load(open("C:/Users/USER/Desktop/Python Projects/TestingPythonVS/Computer vison/Sign Language detector/BinaryData.pickle","rb"))
SaveFile="Computer vison/Sign Language detector/SignClassifiermodel.p"


#Coversion to nump arrays. I experienced an error where it failed to convert if the dimensions are not exactly the same but I'll worry about that later
data=numpy.asarray(data_Dict["data"])
labels=numpy.asarray(data_Dict["Label"])

#Now we're splitting train and test data.
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

#This is basically just a bunch of decision trees and is the machine learning algorithm I used. You're free to try others.
model=RandomForestClassifier()

model.fit(x_train,y_train)

LabelPrediction=model.predict(x_test)

#Accuracy test.
score=accuracy_score(LabelPrediction,y_test)



#It should print about 1 or extremely close to since it is jsust a static image being slightly change for a video.
print(f"This is the score {score}")

with open(SaveFile,"wb") as myFile:
    pickle.dump({"model":model},myFile)



