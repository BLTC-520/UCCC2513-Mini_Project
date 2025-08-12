#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include "Supp.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

Ptr<ml::StatModel> SVMmodel;
Ptr<ml::StatModel> ForestModel;

string classifyShape(const vector<Point>& contour);
String predictAndShowMeaning(const string&, const string&, const Mat);
string getClassMeaning(int);

int calculateCoverage(Mat source);

////////////////////////////////////// Segmentation part/////////////////////////////////////////////////////////////////////
void generateContour(Mat originalImage, Mat processedImage, Mat& outputImage, Mat& outputMask, Mat& segmentedImage) {

    Point2i		center;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(processedImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    // Draw contours on the original image
    Mat contourImage = originalImage.clone();
    drawContours(contourImage, contours, -1, Scalar(0, 255, 0), 1); // Draw contours in green
    contourImage.copyTo(outputImage);


    Mat canvasColor, canvasGray;
    canvasColor.create(originalImage.rows, originalImage.cols, CV_8UC3),
        canvasGray.create(originalImage.rows, originalImage.cols, CV_8U);

    int				index = 0, max = 0; // used to record down the longest contour

    for (int i = 0; i < contours.size(); i++) { // We could have more than one sign in image
        canvasGray = 0;
        if (max < contours[i].size()) { // Find the longest contour as sign boundary
            max = contours[i].size();
            index = i;
        }
        drawContours(canvasColor, contours, i, Scalar(0, 255, 0)); // draw boundaries
        drawContours(canvasGray, contours, i, 255);

        // The code below compute the center of the region
        Moments M = moments(canvasGray);
        center.x = M.m10 / M.m00;
        center.y = M.m01 / M.m00;

        floodFill(canvasGray, center, 255); // fill inside sign boundary
    }
    canvasGray = 0;
    drawContours(canvasGray, contours, index, 255);

    Moments M = moments(canvasGray);
    center.x = M.m10 / M.m00;
    center.y = M.m01 / M.m00;

    // generate mask of the sign
    if (center.x > 0 && center.y > 0) { //Ensure the center is found
        floodFill(canvasGray, center, 255); // fill inside sign boundary
        cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);
        canvasGray.copyTo(outputMask);

        canvasColor = canvasGray & originalImage;
        canvasColor.copyTo(segmentedImage);
    }
    else {
        canvasGray.copyTo(outputMask);
        originalImage.copyTo(segmentedImage);
    }

}


Mat RedSegmentation(Mat win[], int& winCount, Mat legends[], Mat deduct) {
    Mat redMask, mask;

    //Get the average red color
    int avg, pixels = deduct.rows * deduct.cols;
    Scalar deductSum = cv::sum(deduct);
    avg = deductSum[2] / pixels;

    if (avg < 10) deduct *= 2; // brighten the image if too low


    Mat hsv, lowerRed;
    cvtColor(deduct, hsv, COLOR_BGR2HSV);


    Scalar redLower(0, 0, 35);
    Scalar redUpper(0, 255, 165);

    inRange(hsv, redLower, redUpper, lowerRed); //Binary mask for red color in range

    cvtColor(lowerRed, lowerRed, COLOR_GRAY2BGR); //Binary mask to BGR


    // Split the image into color channels
    std::vector<cv::Mat> colorChannels;
    split(lowerRed, colorChannels);


    // Get the red channel (colorChannels[2] is the red channel)
    // Convert red channel to BGR format (just to visualize)
    cvtColor(colorChannels[2], redMask, COLOR_GRAY2BGR);
    //putText(legends[winCount], "Red mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //redMask.copyTo(win[winCount++]);


    // Threshold the red channel to create a binary mask
    threshold(colorChannels[2], mask, 15, 255, THRESH_BINARY);

    return mask;
}


Mat BlueSegmentation(Mat win[], int& winCount, Mat legends[], Mat deduct) {
    Mat blueMask, mask;

    //Get the average red color
    int avg, pixels = deduct.rows * deduct.cols;
    Scalar deductSum = cv::sum(deduct);
    avg = deductSum[0] / pixels; //Blue channel

    if (avg < 10) deduct *= 5; // brighten the image if too low


    Mat hsv, lowerBlue;
    cvtColor(deduct, hsv, COLOR_BGR2HSV);

    // Define the blue color range in HSV
    Scalar blueLower(100, 90, 35);   // Lower bound for blue
    Scalar blueUpper(140, 255, 255); // Upper bound for blue

    inRange(hsv, blueLower, blueUpper, lowerBlue); //Binary mask for red color in range

    cvtColor(lowerBlue, lowerBlue, COLOR_GRAY2BGR); //Binary mask to BGR


    // Split the image into color channels
    std::vector<cv::Mat> colorChannels;
    split(lowerBlue, colorChannels);


    // Get the red channel (colorChannels[2] is the red channel)
    // Convert red channel to BGR format (just to visualize)
    cvtColor(colorChannels[2], blueMask, COLOR_GRAY2BGR);
    //putText(legends[winCount], "Blue mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //blueMask.copyTo(win[winCount++]);


    // Threshold the red channel to create a binary mask
    threshold(colorChannels[2], mask, 15, 255, THRESH_BINARY);

    return mask;
}


Mat YellowSegmentation(Mat win[], int& winCount, Mat legends[], Mat deduct) {
    Mat hsv, yellowMask;
    cvtColor(deduct, hsv, COLOR_BGR2HSV);

    // Define the blue color range in HSV
    Scalar yellowLower(14, 120, 0);
    Scalar yellowUpper(38, 255, 255);

    inRange(hsv, yellowLower, yellowUpper, yellowMask); //Binary mask for red color in range

    cvtColor(yellowMask, yellowMask, COLOR_GRAY2BGR); //Binary mask to BGR

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(yellowMask, yellowMask, MORPH_CLOSE, kernel);
    morphologyEx(yellowMask, yellowMask, MORPH_OPEN, kernel);


    //putText(legends[winCount], "Yellow Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //yellowMask.copyTo(win[winCount++]);

    cvtColor(yellowMask, yellowMask, COLOR_BGR2GRAY);

    return yellowMask;
}

int calculateCoverage(Mat source) {
    // Convert the image to grayscale
    cv::Mat grayImage;

    if (source.channels() == 3) {
        cvtColor(source, grayImage, COLOR_BGR2GRAY);
    }
    else {
        source.copyTo(grayImage);
    }

    // Count non-zero pixels in the grayscale image
    int nonZeroCount = cv::countNonZero(grayImage);

    // if higher than 80% its most likely wrong
    int pixels = source.rows * source.cols;
    int coverage = double(nonZeroCount) / double(pixels) * 100;
    return coverage;
}

void segmentation1() {
    int const noCols = 2;
    int const noRows = 1;
    int const heightGap = 15, widthGap = 3;

    String imgPattern("Inputs/Traffic signs/*.png");
    vector<string> imageNames;
    Mat originalImage, processedImage, largeWin, win[noCols * noRows], legends[noCols * noRows];
    Mat deduct, mask;

    // Get all the image names
    cv::glob(imgPattern, imageNames, true);

    for (int i = 0; i < imageNames.size(); i++) {
        originalImage = imread(imageNames[i]);

        createWindowPartition(originalImage, largeWin, win, legends, noRows, noCols);

        int winCount = 0;
        putText(legends[winCount], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        originalImage.copyTo(win[winCount++]);

        GaussianBlur(originalImage, processedImage, Size(9, 9), 0, 0); // Apply gaussian blur to reduce noice

        Mat blurredImage;
        processedImage.copyTo(blurredImage);

        cvtColor(originalImage, processedImage, COLOR_BGR2GRAY); // Change to gray scale
        cvtColor(processedImage, processedImage, COLOR_GRAY2BGR); // Change back to rgb so that it is visible

        deduct = originalImage - processedImage;

        //putText(legends[winCount], "Deduction", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        //deduct.copyTo(win[winCount++]);



        Mat erode1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 3));
        Mat erode2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        Mat dilate1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(6, 4));

        Mat masks[3];
        masks[0] = RedSegmentation(win, winCount, legends, deduct);
        masks[1] = BlueSegmentation(win, winCount, legends, deduct);
        masks[2] = YellowSegmentation(win, winCount, legends, blurredImage);

        Mat finalSegmentedOutput;
        int finalCoverage = 0;

        for (int j = 0; j < 3; j++) {
            Mat finalMask;
            if (j != 2) { // yellow segmentation doesnt require erosion
                cv::erode(masks[j], finalMask, erode1);
            }
            else {
                finalMask = masks[j];
            }

            Mat contourOutput, generatedMask, segmentedImage;
            generateContour(originalImage, finalMask, contourOutput, generatedMask, segmentedImage);

            //if more than 80% we assume too must corrosion
            //if lesser than 10% most likely wrong color segmentation
            //if near 100% also wrong
            int coverage = calculateCoverage(generatedMask);

            if (coverage >= 80 && j != 2) {
                cv::erode(masks[j], finalMask, erode2); // erode lesser
                cv::dilate(finalMask, finalMask, dilate1); // Dilate also if the coverage is more than 80% as it most likely is wrong contour
                medianBlur(finalMask, finalMask, 1); // blur it abit

                generateContour(originalImage, finalMask, contourOutput, generatedMask, segmentedImage);

                coverage = calculateCoverage(generatedMask);
            }


            //putText(legends[winCount], "Contour" + to_string(i), Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
            //contourOutput.copyTo(win[winCount++]);

            if (coverage > 10 && coverage < 90) {
                if (max(finalCoverage, coverage) == coverage) {
                    finalCoverage = coverage;
                    segmentedImage.copyTo(finalSegmentedOutput);
                }
            }
        }

        if (finalCoverage == NULL) { // If no assignment
            originalImage.copyTo(finalSegmentedOutput);
        }
        String name = predictAndShowMeaning("svm_modelHog.yml", "SVM", originalImage);

        putText(legends[winCount], name, Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        finalSegmentedOutput.copyTo(win[winCount++]);
        cout << imageNames[i];

        // Display the large window with sub-windows
        imshow("Classified Image", largeWin);

        waitKey(0); // Wait for a key press

        destroyAllWindows(); //Close the window
    }


}
void segmentation2() {
    int const noCols = 2;
    int const noRows = 1;
    int const heightGap = 15, widthGap = 3;

    String imgPattern("Inputs/Traffic signs/*.png");
    vector<string> imageNames;
    Mat originalImage, processedImage, largeWin, win[noCols * noRows], legends[noCols * noRows];
    Mat deduct, mask;

    // Get all the image names
    cv::glob(imgPattern, imageNames, true);
    cout << imageNames.size() << endl;

    for (int i = 0; i < imageNames.size(); i++) {
        originalImage = imread(imageNames[i]);


        createWindowPartition(originalImage, largeWin, win, legends, noRows, noCols);

        int winCount = 0;
        putText(legends[winCount], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        originalImage.copyTo(win[winCount++]);

        GaussianBlur(originalImage, processedImage, Size(9, 9), 0, 0); // Apply gaussian blur to reduce noice

        Mat blurredImage;
        processedImage.copyTo(blurredImage);

        cvtColor(originalImage, processedImage, COLOR_BGR2GRAY); // Change to gray scale
        cvtColor(processedImage, processedImage, COLOR_GRAY2BGR); // Change back to rgb so that it is visible

        deduct = originalImage - processedImage;

        //putText(legends[winCount], "Deduction", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        //deduct.copyTo(win[winCount++]);



        Mat erode1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 3));
        Mat erode2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        Mat dilate1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(6, 4));

        Mat masks[3];
        masks[0] = RedSegmentation(win, winCount, legends, deduct);
        masks[1] = BlueSegmentation(win, winCount, legends, deduct);
        masks[2] = YellowSegmentation(win, winCount, legends, blurredImage);

        Mat finalSegmentedOutput;
        int finalCoverage = 0;

        for (int j = 0; j < 3; j++) {
            Mat finalMask;
            if (j != 2) { // yellow segmentation doesnt require erosion
                cv::erode(masks[j], finalMask, erode1);
            }
            else {
                finalMask = masks[j];
            }

            Mat contourOutput, generatedMask, segmentedImage;
            generateContour(originalImage, finalMask, contourOutput, generatedMask, segmentedImage);

            //if more than 80% we assume too must corrosion
            //if lesser than 10% most likely wrong color segmentation
            //if near 100% also wrong
            int coverage = calculateCoverage(generatedMask);

            if (coverage >= 80 && j != 2) {
                cv::erode(masks[j], finalMask, erode2); // erode lesser
                cv::dilate(finalMask, finalMask, dilate1); // Dilate also if the coverage is more than 80% as it most likely is wrong contour
                medianBlur(finalMask, finalMask, 1); // blur it abit

                generateContour(originalImage, finalMask, contourOutput, generatedMask, segmentedImage);

                coverage = calculateCoverage(generatedMask);
            }


            //putText(legends[winCount], "Contour" + to_string(i), Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
            //contourOutput.copyTo(win[winCount++]);

            if (coverage > 10 && coverage < 90) {
                if (max(finalCoverage, coverage) == coverage) {
                    finalCoverage = coverage;
                    segmentedImage.copyTo(finalSegmentedOutput);
                }
            }
        }

        if (finalCoverage == NULL) { // If no assignment
            originalImage.copyTo(finalSegmentedOutput);
        }
        String name = predictAndShowMeaning("randomForest_modelHOG.yml", "RTrees", originalImage);
        putText(legends[winCount], name, Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
        finalSegmentedOutput.copyTo(win[winCount++]);


        // Display the large window with sub-windows
        imshow("Classified Image", largeWin);

        waitKey(0); // Wait for a key press

        destroyAllWindows(); //Close the window
    }
}

////////////////////////////////////// Feature extraction part/////////////////////////////////////////////////////////////////////
// Function to extract HOG features from an image
void extractHOGFeatures(const Mat& image, vector<float>& features, const Size& resizeSize = Size(64, 64)) {
    Mat resizedImage, grayImage;
    resize(image, resizedImage, resizeSize);
    cvtColor(resizedImage, grayImage, COLOR_BGR2GRAY);

    HOGDescriptor hog(
        resizeSize,  // Window size
        Size(16, 16),  // Block size
        Size(8, 8),    // Block stride
        Size(8, 8),    // Cell size
        9              // Number of bins
    );

    hog.compute(grayImage, features);
}

// Function to write HOG features to CSV
void writeHOGFeaturesToCSV(const string& csvPath, const vector<string>& imageNames) {
    ofstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        cerr << "Error: Could not open CSV file for writing." << endl;
        return;
    }

    // Process the first image to determine the number of features
    Mat firstImage = imread(imageNames[0]);
    if (firstImage.empty()) {
        cerr << "Error: Could not open or find the first image " << imageNames[0] << endl;
        return;
    }
    cout << "Extracting features...... Please wait for a while" << endl;
    vector<float> sampleFeatures;
    extractHOGFeatures(firstImage, sampleFeatures);

    // Write CSV header
    csvFile << "filename,label";
    for (size_t i = 0; i < sampleFeatures.size(); ++i) {
        csvFile << ",feature_" << i;
    }
    csvFile << endl;

    // Process all images
    for (const string& imagePath : imageNames) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Could not open or find the image " << imagePath << endl;
            continue;
        }

        resize(image, image, Size(64, 64)); //keep all sizes same

        vector<float> features;
        extractHOGFeatures(image, features);

        string filename = imagePath.substr(imagePath.find_last_of("/\\") + 1);
        string labelStr = filename.substr(0, 3);  // Extract first 3 characters for label
        int label = stoi(labelStr);

        // Write data to CSV
        csvFile << filename << "," << label;
        for (float feature : features) {
            csvFile << "," << feature;
        }
        csvFile << endl;
    }

    csvFile.close();
    cout << "Feature extraction complete and saved to " << csvPath << endl;
}

// Function to extract color histogram features from an image
void extractColorHistogram(const Mat& image, vector<float>& features) {
    // Convert image to HSV color space
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Define histogram parameters
    int histSize[] = { 5, 5, 5 }; // Number of bins for each channel (Hue, Saturation, Value)
    float hRange[] = { 0, 256 }; // Range of values for each channel
    const float* hRangeList[] = { hRange, hRange, hRange }; // Range for each channel

    // Compute histograms for HSV channels
    Mat hist;
    int channels[] = { 0, 1, 2 }; // Channels for Hue, Saturation, Value
    calcHist(&hsvImage, 1, channels, Mat(), hist, 3, histSize, hRangeList, true, false);

    // Normalize the histogram
    hist.convertTo(hist, CV_32F); // Ensure histogram is in float format
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    // Flatten the histogram to a vector of features
    features.clear();
    int binCount = histSize[0] * histSize[1] * histSize[2];
    features.resize(binCount);

    // Access histogram data correctly
    for (int i = 0; i < histSize[0]; ++i) {
        for (int j = 0; j < histSize[1]; ++j) {
            for (int k = 0; k < histSize[2]; ++k) {
                int idx = i + histSize[0] * (j + k * histSize[1]);
                features[idx] = hist.at<float>(i, j, k);
            }
        }
    }
}

void generateConfusionMatrix(const Mat& trueLabels, const Mat& predictedLabels, Mat& confusionMatrix) {
    int numLabels = 3; // Replace with the actual number of classes
    confusionMatrix = Mat::zeros(numLabels, numLabels, CV_32S);

    // Convert predictions to integer format if necessary
    Mat tempPredictions;
    predictedLabels.convertTo(tempPredictions, CV_32S);

    // Check dimensions
    if (trueLabels.rows != tempPredictions.rows) {
        cerr << "Error: Number of true labels does not match number of predicted labels." << endl;
        return;
    }

    // Populate confusion matrix
    for (int i = 0; i < trueLabels.rows; ++i) {
        int trueLabel = trueLabels.at<int>(i, 0);
        int predictedLabel = tempPredictions.at<int>(i, 0);

        if (trueLabel >= 0 && trueLabel < numLabels && predictedLabel >= 0 && predictedLabel < numLabels) {
            confusionMatrix.at<int>(trueLabel, predictedLabel)++;
        }
        else {
            cerr << " True label: " << trueLabel << ", Predicted label: " << predictedLabel << endl;
        }
    }

    // Print the confusion matrix
    cout << "Confusion Matrix:" << endl;
    for (int i = 0; i < numLabels; ++i) {
        for (int j = 0; j < numLabels; ++j) {
            cout << confusionMatrix.at<int>(i, j) << " ";
        }
        cout << endl;
    }
}

void HOG() {
    string folderPath = "Inputs/tsrd-train/";  // Replace with your folder path
    vector<string> imageNames;
    glob(folderPath + "*.png", imageNames, true);

    if (imageNames.empty()) {
        cerr << "Error: No images found in the directory: " << folderPath << endl;
        return;
    }

    string csvPath = "HOGFeatures.csv";
    writeHOGFeaturesToCSV(csvPath, imageNames);

}
// Function to load features and labels from CSV

void ColorHOG() {
    // Folder path for images
    string folderPath = "Inputs/tsrd-train/"; // Replace with your folder path
    vector<string> imageNames;
    glob(folderPath + "*.png", imageNames, true);

    // Open CSV file for writing
    ofstream csvFile("ColorHistogramFeatures.csv");
    if (!csvFile.is_open()) {
        cerr << "Error: Could not open CSV file for writing." << endl;
        return;
    }
    cout << "Extracting features...... Please wait for a while" << endl;
    // Write CSV header
    csvFile << "filename,label,feature_0,feature_1,...,feature_N" << endl;

    for (const string& imagePath : imageNames) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Could not open or find the image " << imagePath << endl;
            continue;
        }

        resize(image, image, Size(64, 64));

        vector<float> features;
        extractColorHistogram(image, features);

        // Extract filename from path
        string filename = imagePath.substr(imagePath.find_last_of("/\\") + 1);

        // Extract label from the first 3 digits of the filename
        string labelStr = filename.substr(0, 3); // Extract first 3 characters
        int label = stoi(labelStr); // Convert to integer

        // Write features to CSV
        csvFile << filename << "," << label;
        for (float feature : features) {
            csvFile << "," << feature;
        }
        csvFile << endl;
    }

    csvFile.close();
    cout << "Feature extraction complete and saved to ColorHistogramFeatures.csv" << endl;

}



//////////////////////////////////// Classification part /////////////////////////////////////////////////////////////////////
// Function to load features and labels from CSV
void loadCSV(const string& filename, Mat& features, Mat& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    string line;

    // Skip header line
    getline(file, line);

    vector<float> rowFeatures;
    vector<int> rowLabels;

    while (getline(file, line)) {
        stringstream ss(line);
        string item;

        // Read filename (skip it)
        getline(ss, item, ',');

        // Read label
        int label;
        if (!(getline(ss, item, ',') && istringstream(item) >> label)) {
            cerr << "Error: Could not parse label from line: " << line << endl;
            continue; // Skip this line if label parsing fails
        }
        rowLabels.push_back(label);

        // Read features
        rowFeatures.clear();
        while (getline(ss, item, ',')) {
            try {
                float feature = stof(item);
                rowFeatures.push_back(feature);
            }
            catch (const invalid_argument& e) {
                cerr << "Error: Could not parse feature from line: " << line << endl;
                rowFeatures.clear();
                break; // Skip this line if feature parsing fails
            }
        }

        // Convert rowFeatures to a Mat if it is not empty
        if (!rowFeatures.empty()) {
            Mat row(rowFeatures);
            features.push_back(row.t());
        }
    }

    // Convert labels vector to Mat
    labels = Mat(rowLabels, true).reshape(1, rowLabels.size());
    labels.convertTo(labels, CV_32S); // Ensure labels are in correct format
    features.convertTo(features, CV_32F); // Ensure features are in correct format

    // Debug: Print the size of the features and labels
    cout << "Loaded features: " << features.rows << "x" << features.cols << endl;
    cout << "Loaded labels: " << labels.rows << "x" << labels.cols << endl;
}

// Function to shuffle and split data
void shuffleAndSplit(const Mat& features, const Mat& labels, Mat& trainFeatures, Mat& trainLabels, Mat& testFeatures, Mat& testLabels, float trainRatio) {
    // Create an index vector
    vector<int> indices(features.rows);
    iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., features.rows-1

    // Shuffle indices
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    // Split indices
    int trainSize = static_cast<int>(trainRatio * features.rows);

    // Split data
    trainFeatures.create(trainSize, features.cols, features.type());
    trainLabels.create(trainSize, labels.cols, labels.type());
    testFeatures.create(features.rows - trainSize, features.cols, features.type());
    testLabels.create(features.rows - trainSize, labels.cols, labels.type());

    for (int i = 0; i < features.rows; ++i) {
        if (i < trainSize) {
            features.row(indices[i]).copyTo(trainFeatures.row(i));
            labels.row(indices[i]).copyTo(trainLabels.row(i));
        }
        else {
            features.row(indices[i]).copyTo(testFeatures.row(i - trainSize));
            labels.row(indices[i]).copyTo(testLabels.row(i - trainSize));
        }
    }
}

void SVMClassification() {
    // Load all data
    Mat allFeatures, allLabels;
    loadCSV("HOGFeatures.csv", allFeatures, allLabels);

    // Shuffle and split data
    Mat trainFeatures, trainLabels, testFeatures, testLabels;
    shuffleAndSplit(allFeatures, allLabels, trainFeatures, trainLabels, testFeatures, testLabels, 0.80f);

    // Define SVM parameters
    Ptr<SVM> svm = SVM::create();
    svm->setKernel(SVM::LINEAR); // or SVM::RBF for non-linear
    svm->setC(1); // Regularization parameter
    svm->setGamma(0.5); // Used if kernel is RBF
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    // Train SVM
    svm->train(trainFeatures, ROW_SAMPLE, trainLabels);
    svm->save("svm_modelHOG.yml");

    cout << "SVM model trained and saved to svm_modelHOG.yml" << endl;


    // Predict and evaluate
    Mat predictions;
    Ptr<SVM> loadedSVM = SVM::load("svm_modelHOG.yml");
    if (loadedSVM.empty()) {
        cerr << "Error: Could not load the SVM model." << endl;
        return;
    }

    loadedSVM->predict(testFeatures, predictions);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        cout << "Test Label: " << testLabels.at<int>(i, 0) << "   Prediction Label: " << static_cast<int>(predictions.at<float>(i, 0)) << endl;
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictions.at<float>(i, 0))) {
            ++correct;
        }
    }

    float accuracy = static_cast<float>(correct) / testLabels.rows;
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;

    // Generate and display the confusion matrix
    Mat confusionMatrix;
    generateConfusionMatrix(testLabels, predictions, confusionMatrix);

}

void RandomForest() {
    // Load all data
    Mat allFeatures, allLabels;
    loadCSV("HOGFeatures.csv", allFeatures, allLabels);

    // Shuffle and split data
    Mat trainFeatures, trainLabels, testFeatures, testLabels;
    shuffleAndSplit(allFeatures, allLabels, trainFeatures, trainLabels, testFeatures, testLabels, 0.99f);

    // Define Random Forest parameters
    Ptr<RTrees> randomForest = RTrees::create();
    randomForest->setMaxDepth(10); // Maximum depth of the tree
    randomForest->setMinSampleCount(2); // Minimum number of samples required at a leaf node
    randomForest->setRegressionAccuracy(0); // Regression accuracy: N/A here
    randomForest->setUseSurrogates(false); // Use surrogate splits
    randomForest->setMaxCategories(10); // Maximum number of categories (useful for categorical data)
    randomForest->setPriors(Mat()); // The prior probabilities of the classes
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); // Termination criteria

    // Train Random Forest
    randomForest->train(trainFeatures, ROW_SAMPLE, trainLabels);
    randomForest->save("randomForest_modelHOG.yml");

    cout << "Random Forest model trained and saved to randomForest_modelHOG.yml" << endl;

    // Predict and evaluate
    Mat predictions;
    Ptr<RTrees> loadedRandomForest = RTrees::load("randomForest_modelHOG.yml");
    if (loadedRandomForest.empty()) {
        cerr << "Error: Could not load the Random Forest model." << endl;
        return;
    }

    loadedRandomForest->predict(testFeatures, predictions);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        cout << "Test Label: " << testLabels.at<int>(i, 0) << "   Prediction Label: " << static_cast<int>(predictions.at<float>(i, 0)) << endl;
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictions.at<float>(i, 0))) {
            ++correct;
        }
    }

    Mat confusionMatrix;
    generateConfusionMatrix(testLabels, predictions, confusionMatrix);


    float accuracy = static_cast<float>(correct) / testLabels.rows;
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;

}



//////////////////////////////////// Display part /////////////////////////////////////////////////////////////////////
// Function to get the name of the class 
string getClassMeaning(int classNumber) {
    map<int, string> classMeanings = {
        {0, "Speed limit (5km/h)"},
        {1, "Speed limit (15km/h)"},
        {2, "Speed limit (30km/h)"},
        {3, "Speed limit (40km/h)"},
        {4, "Speed limit (50km/h)"},
        {5, "Speed limit (60km/h)"},
        {6, "Speed limit (70km/h)"},
        {7, "Speed limit (80km/h)"},
        {8, "No straight&turn left"},
        {9, "No straight&turn right"},
        {10, "No go straight"},
        {11, "No turn left"},
        {12, "No turn left&right"},
        {13, "No turn right"},
        {14, "No overtaking"},
        {15, "No U-turn"},
        {16, "No vehicle allowed"},
        {17, "No horn allowed"},
        {18, "Speed limit (40km/h)"},
        {19, "Speed limit (50km/h)"},
        {20, "Go straight or turn right"},
        {21, "Go straight"},
        {22, "Turn left ahead"},
        {23, "Turn left &right ahead"},
        {24, "Turn right ahead"},
        {25, "Keep left"},
        {26, "Keep right"},
        {27, "Roundabout"},
        {28, "Only car allowed"},
        {29, "Sound horn sign"},
        {30, "Bicycle lane"},
        {31, "U-turn sign"},
        {32, "Road divides"},
        {33, "Traffic light sign"},
        {34, "Warning sign"},
        {35, "Pedestrian crossing symbol"},
        {36, "Bicycle traffic warning"},
        {37, "School crossing sign"},
        {38, "Sharp bend"},
        {39, "Sharp bend"},
        {40, "Danger steep hill ahead warning"},
        {41, "Danger steep hill ahead warning"},
        {42, "Slowing sign"},
        {43, "T-junction ahead"},
        {44, "T-junction ahead"},
        {45, "Village warning sign"},
        {46, "Snake road"},
        {47, "Railroad level crossing sign"},
        {48, "Under construction"},
        {49, "Snake road"},
        {50, "Railroad level crossing sign"},
        {51, "Accident frequent happened sign"},
        {52, "Stop"},
        {53, "No entry"},
        {54, "No Stopping"},
        {55, "No entry"},
        {56, "Give way"},
        {57, "Stop for checking purpose"}
    };

    if (classMeanings.find(classNumber) != classMeanings.end()) {
        return classMeanings[classNumber];
    }
    else {
        return "Unknown traffic sign";
    }
}

void showWindow(Mat src, Mat seg, String class_name) {
    int const	noOfImagePerCol = 1, noOfImagePerRow = 2;
    Mat			largeWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
    createWindowPartition(src, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);
    src.copyTo(win[0]);
    putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[1], class_name, Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    seg.copyTo(win[1]);
    imshow("Window", largeWin);
    waitKey(0);
    destroyAllWindows();
}

String predictAndShowMeaning(const string& modelPath, const string& modelType, const Mat inputImage) {
    string folderPath = "Inputs/Traffic signs/"; // Folder containing the images
    Ptr<ml::StatModel> model;
    Mat image;
    inputImage.copyTo(image);

    // Load the appropriate model based on modelType
    if (modelType == "SVM") {
        if (SVMmodel == NULL) {
            cout << "Loading SVM model .....";
            SVMmodel = StatModel::load<SVM>(modelPath);
            cout << "SVM model loaded!" << endl;
        }
        model = SVMmodel;

    }
    else if (modelType == "RTrees") {
        model = StatModel::load<RTrees>(modelPath);
    }
    else {
        cerr << "Error: Unsupported model type." << endl;
        return "Error model type";
    }

    // Check if model is loaded correctly
    if (model.empty()) {
        cerr << "Error: Failed to load the model from: " << modelPath << endl;
        return "Error loading model";
    }


    if (image.empty()) {
        cerr << "Error: Failed to load image: " << endl;
        return "Error loading image";
    }

    resize(image, image, Size(64, 64));
    // Extract features based on the chosen model type
    vector<float> features;

    if (modelType == "SVM") {
        //extractColorHistogram(image, features);
        extractHOGFeatures(image, features);
    }
    else if (modelType == "RTrees") {
        extractHOGFeatures(image, features);
    }

    // Convert features to Mat
    Mat featureMat(1, (int)features.size(), CV_32F, features.data());

    // Predict using the loaded model
    Mat predictions;
    model->predict(featureMat, predictions);

    int classNumber = static_cast<int>(predictions.at<float>(0));

    // Show the image and print the predicted class meaning
    string className = getClassMeaning(classNumber);
    cout << "Image Class: " << classNumber << " (" << className << ")" << endl;
    return className;
}

int main() {
    int choice = -1;  // Initialize choice to an invalid value

    while (choice != 0) {
        cout << "Please choose number for the following task." << endl;
        cout << "1. Segmentation of traffic sign (HOG + SVM) " << endl;
        cout << "2. Segmentation of traffic sign (colorHOG + Random Forest) " << endl;
        cout << "3. HOG Extraction and SVM classification model training" << endl;
        cout << "4. Color HOG Extraction and Random Forest model training" << endl;
        cout << "0. Exit" << endl;
        cout << "Enter your choice: ";

        // Check if input is an integer
        cin >> choice;
        if (cin.fail()) {
            cout << "Invalid input. Please input integer only." << endl << endl;
            cin.clear();  // Clear the error state
            cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Ignore the rest of the invalid input
            choice = -1;
            continue;  // Skip to the next iteration of the loop
        }

        if (choice == 1) {
            /// traffic sign segmentation of (SVM)
            segmentation1();
        }
        else if (choice == 2) {
            /// traffic sign segmentation of (Random Forest)
            segmentation2();
        }
        else if (choice == 3) {

            // Histogram extraction and SVM training

            HOG();
            SVMClassification();
            main();
        }
        else if (choice == 4) {

            // HOG extraction and Random Forest training

            ColorHOG();
            RandomForest();
            main();
        }
        else if (choice == 0) {
            cout << "Exiting the program..." << endl;
        }
        else {
            cout << "Invalid input. Please choose a valid option." << endl << endl;
        }
    }

    return 0;
}