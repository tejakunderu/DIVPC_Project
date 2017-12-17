#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2\core\core.hpp"
#include "opencv2\highgui.h"
#include "opencv2/highgui.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv\ml.h"
#include "opencv2\ml.hpp"
#include "opencv2\ml\ml.hpp"
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::ml;
using namespace cv::xfeatures2d;

int mainobjsift() {

	// Initialising array for filenames
	char filename[200];

	// Initialising a SIFT descriptor, descriptor matcher
	Ptr<Feature2D> f2d = SIFT::create();
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	// Initialising Bag of words image descriptor extractor 
	// to match keypoints and compute descriptors for new images
	BOWImgDescriptorExtractor bowDE(f2d, matcher);

	// Initialising other variables that would be used in the code
	Mat image, descriptor, featuresUnclustered, dictionary;
	vector<KeyPoint> keypoints;

	vector<vector<float>> trainDataArr;

	Mat bowDescriptor, img, labels;
	Mat trainData(200, 200, CV_32FC1);

	// Every image in each of the 4 categories are read from their respective category folders
	for (int cls = 1; cls <= 4; cls++) {
		for (int f = 1; f < 51; f++) {
			if (f < 10)
				sprintf_s(filename, "4_ObjectCategories2/%i/image_000%i.jpg", cls, f);
			else sprintf_s(filename, "4_ObjectCategories2/%i/image_00%i.jpg", cls, f);

			image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

			// The keypoints and descriptors for the images are calculated and added to a matrix
			f2d->detect(image, keypoints);
			f2d->compute(image, keypoints, descriptor);
			featuresUnclustered.push_back(descriptor);

			printf("Class %i image %i done\n", cls, f);
		}
	}

	// Number of clusters and other kmeans parameters are initialised
	int dictionarySize = 200;
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	
	printf("K means clustering\n");

	// KMeans trainer is used to cluster the descriptors generated in the previous step
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	dictionary = bowTrainer.cluster(featuresUnclustered);
	
	// The calculated cluster data is stored using filestorage making it easier
	// to build the code without having to run the entirety of it
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();

	// Retreiving the stored data from the file storage and back into the matrix
	FileStorage fs1("dictionary.yml", FileStorage::READ);
	fs1["vocabulary"] >> dictionary;
	fs1.release();

	// Setting the vocabulary generated earlier
	bowDE.setVocabulary(dictionary);

	// For every image for each category, the keypoints are detected and used to 
	// calculate BOWDescriptors from matching with the data in the vocabulary
	for (int cls = 1; cls <= 4; cls++) {
		for (int f = 1; f < 51; f++) {
			if (f < 10)
				sprintf_s(filename, "4_ObjectCategories2/%i/image_000%i.jpg", cls, f);
			else sprintf_s(filename, "4_ObjectCategories2/%i/image_00%i.jpg", cls, f);

			img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

			f2d->detect(img, keypoints);

			bowDE.compute(img, keypoints, bowDescriptor);

			// Obtained descriptor is converted to 32 bit float format for svm 
			bowDescriptor.convertTo(bowDescriptor, CV_32FC1);

			// The descriptor is then added to a 2d array 
			// consisting of all descriptors of all images in the dataset
			vector<float> array;
			if (bowDescriptor.isContinuous()) {
				array.assign((float*)bowDescriptor.datastart, (float*)bowDescriptor.dataend);
			}

			// This array of descriptors is the training data
			trainDataArr.push_back(array);
			// Respective class labels are also added to matrix
			labels.push_back(cls);

			printf("Class %i image %i done\n", cls, f);
		}
	}

	// Labels are converted to integer format for svm
	labels.convertTo(labels, CV_32S);

	// The 2d vector is converted to a Mat format for svm
	for (size_t i = 0; i < 200; i++) {
		for (size_t j = 0; j < 200; j++)
		{
			trainData.at<float>(i, j) = trainDataArr[i][j];
		}
	}

	// training data and labels are stored in the file storage to save computational power
	FileStorage fs2("trainData.yml", FileStorage::WRITE);
	fs2 << "trainData" << trainData;
	fs2.release();

	FileStorage fs3("labels.yml", FileStorage::WRITE);
	fs3 << "labels" << labels;
	fs3.release();

	printf("Storing data\n");

	// Initialising svm with multi class type and linear boundary
	// Other terminal criteria are also initialised
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	printf("Training SVM\n");

	// TrainData is created using the features and labels specified in row format
	Ptr<TrainData> td = TrainData::create(trainData, ROW_SAMPLE, labels);
	// training svm
	svm->train(td);

	// setting vocabulary again to start svm classification for new images
	bowDE.setVocabulary(dictionary);

	// Initialising varialbes
	int correctPredictions = 0;
	float totalPredictions = 200;
	float classPredictions = 50;
	float accuracy;

	// Individual class accuracies
	vector<float> class_accuracy(4, 0);

	// 50 new images from each class are taken. Descriptors of these images are features
	// used to classify them
	for (int cls = 1; cls <= 4; cls++) {
		for (int f = 51; f < 101; f++) {
			if (f < 100)
				sprintf_s(filename, "4_ObjectCategories2/%i/image_00%i.jpg", cls, f);
			else sprintf_s(filename, "4_ObjectCategories2/%i/image_0%i.jpg", cls, f);

			img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

			f2d->detect(img, keypoints);

			bowDE.compute(img, keypoints, bowDescriptor);

			bowDescriptor.convertTo(bowDescriptor, CV_32FC1);

			// Predicting the class based on the descriptors generated
			float result = svm->predict(bowDescriptor);
			result = roundf(result);

			// Incrementing the correct predictions if true
			if (result == cls) {
				correctPredictions++;
				class_accuracy[cls - 1]++;
			}
		}
	}

	// calculating accuracy
	accuracy = correctPredictions / totalPredictions;
	accuracy *= 100;

	// calculating accuracies for individual classes and displaying the results
	for (int i = 0; i < 4; i++) {
		class_accuracy[i] /= classPredictions;
		class_accuracy[i] *= 100;
		cout << "Accuracy of class " << (i + 1) << " is " << class_accuracy[i] << " percent" << endl;
	}

	cout << "Accuracy of the object recognition model is " << accuracy << " percent" << endl;

	return 0;

}