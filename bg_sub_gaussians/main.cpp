/*
* Written (W) 2018 uriel
*/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
//#include <opencv/cxcore.h>
//#include <opencv/cvaux.h>
#include <opencv2/bgsegm.hpp>

using namespace cv;

int main()
{
	// Frame
	Mat frame;

	// Foreground mask (generated by GMM)
	Mat fg;

	// Background
	Mat bg;

	// Number of frames we want to process
	int nFrames = 10000;

	// GMM parameters
	double learning_rate = 0.1;
	int history = 300; // How many previous iterations influence the model at this time step.
	int n_mixtures = 200;
	double background_ratio = 0.001;
	double noise_sigma = 190;

	// Initialization of GMM
	Ptr<BackgroundSubtractor> pGMM;
	//pGMM = createBackgroundSubtractorMOG(history, n_mixtures, background_ratio, noise_sigma);
	pGMM = cv::bgsegm::createBackgroundSubtractorMOG();

	//Ptr<BackgroundSubtractorMOG2> pGMM_2;
	//pGMM = new BackgroundSubtractorMOG2(history, n_mixtures, background_ratio);

	// Open video
	// VideoCapture cap(0); //Webcam
	VideoCapture cap("../data/Video.mp4");

	// Check if the video was correctly opened
	if (!cap.isOpened())
		return 1;

	// Run over the video
	for (int i = 0; i < nFrames; ++i) {

		// Get the frame
		cap >> frame;

		// Apply the background subtraction
		//pGMM->operator()(frame, fg, learning_rate);
		pGMM->apply(frame, fg, learning_rate);
		//pGMM->getBackgroundImage(fg);

		// Show result
		imshow("Original", frame);
		imshow("GMM", fg);
		imshow("Background",bg);

		waitKey(1);
	}

	return 0;
}