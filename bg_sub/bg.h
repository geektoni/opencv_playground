/*
* Written (W) 2018 uriel
*/

#ifndef OPENCV_BG_H
#define OPENCV_BG_H

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

void bg_train(Mat frame, Mat* background);
void bg_update(Mat frame, Mat* background);


#endif //OPENCV_BG_H
