#ifndef Supp_type
#define Supp_type

#include	<opencv2/opencv.hpp>
#include	<opencv2/highgui/highgui.hpp>
#include	<iostream>

#define		SEPH	3
#define		SEPV	15

using namespace cv;
using namespace std;

void createWindowPartition(Mat srcI, Mat &largeWin, Mat win[], Mat legends[], int noOfImagePerCol = 1,
	int noOfImagePerRow = 1, int sepH = SEPH, int sepV = SEPV);
void displayCaption(Mat win, const char* caption, int y=20, int x=6);

	// The following 2 functions can be used in 2 fashions: (1) safe fashion as out = convertGrayFloat2GrayImage(in), or
	// (2) convertGrayFloat2GrayImage(in, out) if out has been assigned memory for suitable image type
Mat convertGrayFloat2GrayImage(Mat grayFloat, Mat *outputImage=NULL);
Mat convertGrayFloat2ColorImage(Mat grayFloat, Mat *outputImage=NULL);
//Mat absDisplay(Mat m1);
Mat generateGaussian(int rows, int cols, int sigma);

#endif