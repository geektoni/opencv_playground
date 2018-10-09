/*
* Written (W) 2018 uriel
*/

#include "bg.h"
#include <iostream>

using namespace std;

static int ctr = 1;

/* Alpha value for ABS */
static float alpha = 0.1;

void bg_train(Mat frame, Mat *background) {
	if (ctr == 1)
	{
		cout << "Initial background storage" << endl;
		frame.copyTo(*background);
	}
	ctr++;
}

/**
 * Adaptive Background Subtraction.
 *
 * @param frame current frame
 * @param background background
 */
void bg_update(Mat frame, Mat *background) {
	Mat new_bg;
	new_bg = frame*alpha+(1.0-alpha)*(*background);
	new_bg.copyTo(*background);
}
