// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void changeGrayAdditive(Mat src,float value) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();

	// The fastest approach of accessing the pixels -> using pointers
	uchar* lpSrc = src.data;
	uchar* lpDst = dst.data;
	int w = (int)src.step; // no dword alignment is done !!!
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			uchar val = lpSrc[i * w + j];
			if (val + value > 255)
				lpDst[i * w + j] = 255;
			else
				lpDst[i * w + j] = val + value;
		}
	imshow("modified by additive factor image", dst);
	waitKey(0);
}


void changeGreyMultiplicative(Mat src, float value) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();

	// The fastest approach of accessing the pixels -> using pointers
	uchar* lpSrc = src.data;
	uchar* lpDst = dst.data;
	int w = (int)src.step; // no dword alignment is done !!!
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			uchar val = lpSrc[i * w + j];
			if (val * value > 255)
				lpDst[i * w + j] = 255;
			else
				lpDst[i * w + j] = val * value;
		}
	imshow("modified by multiplicative factor image", dst);
	imwrite("modifiedMulitplicative.jpg", dst);
	waitKey(0);
}


void color4Squares() {
	Mat squares(256, 256, CV_8UC3);
	int w = (int)squares.step;
	unsigned char vb, vg, vr;
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			if (i < 128 && j < 128) {//white
				vb = 255;
				vg = 255;
				vr = 255;
			}
			else if (i < 128 && j >= 128) {//red
				vb = 0;
				vg = 0;
				vr = 255;
			}
			else if (i >= 128 && j < 128) {//green
				vb = 0;
				vg = 255;
				vr = 0;
			}
			else if (i >= 128 && j >= 128) {//yellow
				vb = 0;
				vg = 255;
				vr = 255;
			}
			Vec3b pixel = Vec3b(0, 0, 0);
			pixel[0] = vb;
			pixel[1] = vg;
			pixel[2] = vr;
			squares.at<Vec3b>(i, j) = pixel;
		}
	}
	imshow("square with squares", squares);
	waitKey(0);
}

void flMatrix3x3() {
	Mat matr(3, 3, CV_32FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			matr.at<float>(i, j) = (float) (rand() % 10 + 1);
		}
	}
	Mat inverse = matr.inv();
	std::cout << "Matrix generated randomly" << std::endl;
	std::cout << matr << std::endl;
	std::cout << "Inverse of the matrix" << std::endl;
	std::cout << inverse;
}

void lab1() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		imshow("input image", src);
		changeGrayAdditive(src, 50.0);
		changeGreyMultiplicative(src, 0.5);
		color4Squares();
		flMatrix3x3();
		//waitKey();
	}
}

void RGB2Grayscale(Mat src) {
	int rs = src.rows;
	int cs = src.cols;
	Mat GS(rs, cs, CV_8UC1);	
	for (int i = 0; i < rs; i++) {
		for (int j = 0; j < cs; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			GS.at<unsigned char>(i, j) = (pixel[0] + pixel[1] + pixel[2]) / 3;
		}
	}
	imshow("ORIGINAL", src);
	imshow("GRAYSCALE", GS);
	waitKey(0);
}

void splitRGB(Mat src) {
	int rs = src.rows;
	int cs = src.cols;
	Mat R(rs, cs, CV_8UC1);
	Mat G(rs, cs, CV_8UC1);
	Mat B(rs, cs, CV_8UC1);
	for (int i = 0; i < rs; i++) {
		for (int j = 0; j < cs; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			B.at<unsigned char>(i, j) = pixel[0];
			G.at<unsigned char>(i, j) = pixel[1];
			R.at<unsigned char>(i, j) = pixel[2];
		}
	}
	imshow("ORIGINAL", src);
	imshow("RED", R);
	imshow("GREEN", G);
	imshow("BLUE", B);
	waitKey(0);
}

void binaryConversion(char fname[]) {
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	int rs = src.rows;
	int cs = src.cols;
	Mat BW(rs, cs, CV_8UC1);
	std::cout << "Input the treshold" << std::endl;
	int tsh;
	std::cin >> tsh;
	for (int i = 0; i < rs; i++) {
		for (int j = 0; j < cs; j++) {
			unsigned char pixel = src.at<unsigned char>(i, j);
			if(pixel >= tsh)
				BW.at<unsigned char>(i, j) = 255;
			else
				BW.at<unsigned char>(i, j) = 0;
		}
	}
	imshow("ORIGINAL GRAYSCALED", src);
	imshow("BINARY", BW);
	waitKey(0);
}

void RGB2HSV(Mat src) {
	int rs = src.rows;
	int cs = src.cols;
	Mat H(rs, cs, CV_8UC1);
	Mat S(rs, cs, CV_8UC1);
	Mat V(rs, cs, CV_8UC1);
	float rn, gn, bn;
	float c;
	float maxrgb, minrgb;
	float hval = 0;
	for (int i = 0; i < rs; i++) {
		for (int j = 0; j < cs; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			bn = (float)pixel[0] / 255;
			gn = (float)pixel[1] / 255;
			rn = (float)pixel[2] / 255;
			maxrgb = max(max(rn, bn), gn);
			minrgb = min(min(rn, bn), gn);
			c = maxrgb - minrgb;
			V.at<unsigned char>(i, j) = (unsigned char)(maxrgb * 255);//value
			if (maxrgb != 0)
				S.at<unsigned char>(i, j) = (unsigned char)(c / maxrgb * 255);//saturation
			else
				S.at<unsigned char>(i, j) = (unsigned char)(0);
			if (c != 0) {
				if (maxrgb == rn)
					hval = 60 * (gn - bn) / c;
				else if (maxrgb == gn)
					hval = 120 + 60 * (bn - rn) / c;
				else if(maxrgb == bn)
					hval = 240 + 60 * (rn - gn) / c;
				if (hval < 0)
					hval = hval + 360;
				H.at<unsigned char>(i, j) = hval * 255 / 360;
			}
			else
				H.at<unsigned char>(i, j) = (unsigned char)(0);
		}
	}
	imshow("ORIGINAL", src);
	imshow("HUE", H);
	imshow("SATURATION", S);
	imshow("VALUE", V);
	waitKey(0);
}

void isInside(Mat src, int i, int j) {

	if (i >= 0 && i <= src.rows && j >= 0 &&  j <= src.cols) {
		std::cout << "Point is inside" << std::endl;
		return;
	}
	std::cout << "Point is not inside" << std::endl;
}

void lab2() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_COLOR);
		splitRGB(src);
		RGB2Grayscale(src);
		binaryConversion(fname);
		RGB2HSV(src);
		int ri, rj;
		std::cout << "Input the row coordinate" << std::endl;
		std::cin >> ri;
		std::cout << "Input the column coordinate" << std::endl;
		std::cin >> rj;
		isInside(src, ri, rj);
	}
}

void computeGrayscaleHistogram(char fname[256], int histogram[256]) {
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	unsigned char collector;
	for (int i = 0; i < 256; i++)
		histogram[i] = 0;
	int rs = src.rows, cs = src.cols;
	for (int i = 0; i < rs; i++) {
		for (int j = 0; j < cs; j++) {
			collector = src.at<unsigned char>(i, j);
			histogram[collector]++;
		}
	}
}

void computeGrayscalePDF(char fname[256], int histogram[256], float pdf[256]) {
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	for (int i = 0; i < 256; i++) {
		pdf[i] = histogram[i] / src.rows * src.cols;
	}
}

void computeGrayscaleHistogramMbins(char fname[256], int *histogram,int bins) {
	Mat src = imread(fname, IMREAD_GRAYSCALE);
	unsigned char collector;
	histogram = (int*)malloc(bins);
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++) {
			collector = src.at<unsigned char>(i, j);
			histogram[(int)(i / bins)] ++;
		}
	}
}

void lab3() {
	char fname[MAX_PATH];
	int grayscaleHistogram[256];
	float grayscalePDF[256];
	int* histogramNBins;
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_COLOR);
		computeGrayscaleHistogram(fname, grayscaleHistogram);
		computeGrayscalePDF(fname, grayscaleHistogram, grayscalePDF);
		showHistogram("GRAYSCALE HISTOGRAM", grayscaleHistogram, 256,200);
		int bins;
		std::cout << "Input the number of bins" << std::endl;
		std::cin >> bins;
		computeGrayscaleHistogramMbins(fname, histogramNBins, bins);
		showHistogram("GRAYSCALE HISTOGRAM M BINS", grayscaleHistogram, bins, 200);
	}
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative\n");
		printf(" 4 - Image negative (fast)\n");
		printf(" 5 - BGR->Gray\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 7 - BGR->HSV\n");
		printf(" 8 - Resize image\n");
		printf(" 9 - Canny edge detection\n");
		printf(" 10 - Edges in a video sequence\n");
		printf(" 11 - Snap frame from live video\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 13 - Lab 1 ops\n");
		printf(" 14 - Lab 2 ops\n");
		printf(" 15 - Lab 3 ops\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testNegativeImage();
				break;
			case 4:
				testNegativeImageFast();
				break;
			case 5:
				testColor2Gray();
				break;
			case 6:
				testImageOpenAndSave();
				break;
			case 7:
				testBGR2HSV();
				break;
			case 8:
				testResize();
				break;
			case 9:
				testCanny();
				break;
			case 10:
				testVideoSequence();
				break;
			case 11:
				testSnap();
				break;
			case 12:
				testMouseClick();
				break;
			case 13:
				lab1();
				break;
			case 14:
				lab2();
				break;
			case 15:
				lab3();
				break;
		}
	}
	while (op!=0);
	return 0;
}