#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {

  // Mat image;
  // image = imread("Google.jpg", 1);

  // namedWindow("Window",1);
  // imshow("Window", image);
  // waitKey(0);

  Mat image, frame, frame_gray, motion_mask, threshold_mask;
  Mat *frames_gray = new Mat[1000];
  VideoCapture cap;
  int nDiff = 10;

  //cap.open("../Video.mp4");

  // Use camera instead
  cap.open(0);

  if (!cap.isOpened())
  	return 0;

  for (int i = 0; i < 1000; i++) {
	  cap >> image;
    cvtColor(image, frame_gray, CV_RGB2GRAY);
    frame_gray.copyTo(frames_gray[i]);
    if (i > nDiff) {
      absdiff(frames_gray[i], frames_gray[i - nDiff], motion_mask);
      threshold(motion_mask, threshold_mask, 50.0, 255, THRESH_BINARY);
      imshow("Motion Mask", motion_mask);
      imshow("Threshold", threshold_mask);
    }

    waitKey(1);
  }

  delete[] frames_gray;
  return 0;
}
