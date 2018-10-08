/*
* Written (W) 2018 uriel
*/

#include "bg.h"
#include <iostream>

using namespace std;

static int ctr = 1;

void bg_train(Mat frame, Mat *background) {
	if (ctr == 1)
	{
		cout << "Initial background storage" << endl;
		frame.copyTo(*background);
	}
	ctr++;
}

void bg_update(Mat frame, Mat *background) {

}
