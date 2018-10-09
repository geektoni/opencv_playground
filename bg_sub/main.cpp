/*
* Written (W) 2018 uriel
*/

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "bg.h"

using namespace cv;

int main()
{
	// Matrixes needed
	Mat frame, frame_gray;
	Mat bg, motion_mask, motion_threshold;

	// Max number of frames
	int nFrames = 10000;

	// Threshold
	int thresh = 50;

	// Video capture object
	VideoCapture cap(0);

	// Check if cap was opened correctly
	if (!cap.isOpened())
	{
		return 1;
	}

	// Goes around all video's frames
	for (int i = 0; i < nFrames; ++i) {

		// Capture one frame from the video
		cap >> frame;

		// Convert to grayscale the current frame
		cvtColor(frame, frame_gray, CV_RGB2GRAY);

		// Store the first frame as background
		bg_train(frame_gray, &bg);

		// We perform background subtraction
		absdiff(bg, frame_gray, motion_mask);

		// We perform thresholding
		threshold(motion_mask, motion_threshold, thresh, 255, THRESH_BINARY);

		// Background update
		bg_update(frame_gray, &bg);

		imshow("original", frame);
		imshow("background", bg);
		imshow("threshold", motion_threshold);

		waitKey(1);
	}

	return 0;
}