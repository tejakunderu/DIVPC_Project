#include <stdio.h>
#include <iostream>
#include "stdafx.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int mainsift() {
	//Reading image into matrix
	Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);

	//Creating a SIFT descriptor extractor
	Ptr<Feature2D> f2d = SIFT::create();
	vector<KeyPoint> keypoints;
	//Detecting keypoints
	f2d->detect(img, keypoints);

	Mat descriptors;
	//Computing descriptors 
	f2d->compute(img, keypoints, descriptors);

	Mat img_keypoints;
	//Drawing the computed keypoints on the input image
	drawKeypoints(img, keypoints, img_keypoints, Scalar::all(-1),
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//Initialising windows to display results
	namedWindow("Input");
	namedWindow("Keypoints");

	//Displaying images in the windows
	imshow("Input", img);
	imshow("Keypoints", img_keypoints);
	//Waiting for user input to close the windows
	waitKey(0);

	return 0;
}