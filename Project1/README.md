************** SIFT/BRISK implementation ***************

Open the project1.sln file using visual studio.
Please find two files named sift.cpp and brisk.cpp.
The main() function in the code needs to be renamed.
Each of the codes takes the image "Lena.jpg" as input. The file is provided with the project.
The input image is displayed in a window named "input". 
The image with keypoint descriptors marked on it is then displayed on a separate window named "keypoints". 
Press any key to close the windows and exit the application.


*************** Object Recognition *****************

Please find two files named objectRecognitionSift.cpp and objectRecognitionBrisk.cpp.
The main() funtion in the code needs to be renamed to run.
The code takes in 4 categories of images under the folder 4_ObjectCategories2.
The categories and images are taken from the Caltech_101 dataset.
From each of the 4 classes, first 50 images are used for training and the next 50 images are used for testing.

For each image, the keypoints and descriptors are calculated using SIFT/BRISK.
These are then separated into clusters using Kmeans using BOWKmeansTrainer.
The vocabulary created is then used to compute BOWImageDescriptors for every image.
The obtained values are stored in a matrix along with the class of the image.
These are given as input to train the Support Vector Machines algorithm.
The trained model is then used to test the 50 test images.

The accuracies of each class and the total accuracy of the classifier are displayed at the end of the program.

Please note that the whole program might take about 10 minutes to run for each descriptor.