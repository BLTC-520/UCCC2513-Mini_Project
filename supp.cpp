#include	"Supp.h"  

// The function below helps to create a larger window with A X B sub-windows.
// For example, 3X4 sub-windows.  A and B are determined by the input parameters of 
// noOfImagePerCol and noOfImagePerRow respectively. The first input srcI is the
// input image based on which we build a larger window largeWin based on the type
// of srcI.  After creation, the means to access each sub-window is through win[].
// For example, the first sub-window is refered as win[0], the second one is refered 
// as win[1] and so on.  A smaller window is also created under each sub-window for 
// the purpose to put legend.  Each can be accessed similarly as with win[i].
void createWindowPartition(Mat srcI, Mat &largeWin, Mat win[], Mat legends[], int noOfImagePerCol,
	int noOfImagePerRow, int sepH, int sepV) {
	// 1st input: source input image
	// 2nd: the created larger window
	// 3th: means to access each sub window
	// 4th: means to access each legend window
	// 5rd, 6th: Obvious
	// 7th: separating space between 2 images in horizontal direction
	// 8th: separating space between 2 images in vertial direction

	int		rows = srcI.rows, cols = srcI.cols, winI = 0, winsrcI = 0;
	Size	sRXC((cols + sepH)*noOfImagePerRow - sepH, (rows + sepV)*noOfImagePerCol), // Size of new window
		s(cols, sepV); // Size of the legend window

	largeWin = Mat::ones(sRXC, srcI.type()) * 64; // create the new window with background color 64
	for (int i = 0; i<noOfImagePerCol; i++) // create the subwindows
		for (int j = 0; j<noOfImagePerRow; j++)
			win[winI++] = largeWin(Range((rows + sepV)*i, (rows + sepV)*i + rows),
			Range((cols + sepH)*j, (cols + sepH)*j + cols));

	for (int bg = 20, i = 0; i<noOfImagePerCol; i++) // create the legend windows
		for (int j = 0; j<noOfImagePerRow; j++) {
			legends[winsrcI] = largeWin(Range((rows + sepV)*i + rows, (rows + sepV)*(i + 1)),
			Range((cols + sepH)*j, (cols + sepH)*j + cols));
			legends[winsrcI] = Scalar(bg, bg, bg); // paint each in different colors
			bg += 30; // such that we can visually see the division from one to other
			if (bg > 80) bg = 20;
			winsrcI++;
		}
}

void displayCaption(Mat win, const char* caption, int y, int x) {
	putText(win, caption, Point(x, y), 3, //CV_FONT_HERSHEY_COMPLEX,
		0.6, Scalar(255, 255, 255));
}

// The output can be passed back by "return" or "out". Hence, must use "return image".
// When using out, the content is written to location specified by out.
// When using return, the received content will point to the newly created image.
Mat convertGrayFloat2GrayImage(Mat grayFloat, Mat *out) { 
	Mat		image;  

	normalize(grayFloat, image, 0, 255, NORM_MINMAX);
	image.convertTo(image, CV_8U);// it uses the function saturate_cast to convert 
	// negative into zero and overflowed number to 255 according to type CV_8U.
	// Hence, some information can be lost.
	if (out != NULL) {
		image.copyTo(*out);
	}
	return image;
}

// See comment from convertGrayFloat2GrayImage()
Mat convertGrayFloat2ColorImage(Mat grayFloat, Mat *out) {
	Mat		tmp;

	normalize(grayFloat, tmp, 0, 255, NORM_MINMAX);
	tmp.convertTo(tmp, CV_8U); // it uses the function saturate_cast to convert 
	// negative into zero and overflowed number to 255 according to type CV_8U.
	// Hence, some information can be lost. 
	cvtColor(tmp, tmp, COLOR_GRAY2BGR);
	if (out != NULL) {
		tmp.copyTo(*out);
	}
	return tmp;
}

// Generate first 2 1D filters. Then combine them to obtain a 2D filter
Mat generateGaussian(int rows, int cols, int sigma) {
	Mat		gx, gy, g;

	gx = getGaussianKernel(cols, sigma, CV_32F);
	gy = getGaussianKernel(rows, sigma, CV_32F);
	transpose(gx, gx);
	g = gy * gx; // Matrix multiplication is used here different from multiply().
	normalize(g, g, 0, 1, NORM_MINMAX);
	
	// The code below is not needed.  It is just for viewing the 1D masks generated.
//	normalize(gx, gx, 0, 1, NORM_MINMAX);
//	imshow("x", gx);
//	normalize(gy, gy, 0, 1, NORM_MINMAX);
//	imshow("y", gy);
	return g;
}