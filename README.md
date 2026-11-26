# Sign-Language-detector-AI
This is the repository for our ISA STEM competition project.
It works like this, using the DataColletion.py, you collect data(duh) using your camera for each letter.
Then in SignlanguagePreparingDataset.py you prepare the data. You just run it and them its good to go If you get an error that means there may be something wrong with your data coleected like your hand was outside a boundary or something.
After when you go to ClassifierTraining.py ,it (guess what?) trains the classifier. You just need to run it and the accuracy score should show 1.0 or should be around 0.95~1.0.If it is not then that means it needs to become more accurate.
Then you go to InferenceClassifier.py and get the inference from the camera to get the sign language from your hand signs.
