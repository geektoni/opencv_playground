#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


struct mouse_info_struct { int x,y; };
struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;

vector<Point> mousev,kalmanv;

void on_mouse(int event, int x, int y, int flags, void* param) {
	{
		last_mouse = mouse_info;
		mouse_info.x = x;
		mouse_info.y = y;

	}
}

// plot points
#define drawCross( center, color, d )                                 \
line( img, Point( center.x - d, center.y - d ),                \
Point( center.x + d, center.y + d ), color, 2, CV_AA, 0); \
line( img, Point( center.x + d, center.y - d ),                \
Point( center.x - d, center.y + d ), color, 2, CV_AA, 0 )

int main () {

	// Matrix in which to plot the trajectories (coloured)
	Mat img(1000, 1000, CV_8UC3);

	// Our Kalman filter
	// M = [x_m, y_m]
	// S = [x, y, v_x, v_y, a_x, a_y] (we record the velocity in both directions
	// and the acceleration)
	KalmanFilter KF(6,2,0);

	// Matrices that will contain the state
	// and the noise. Here we show two ways to
	// create a matrix containing floats.
	Mat_<float> state(6,1);
	Mat processNoise(6,1, CV_32F);

	// Contains the position of the mouse. Kalman does
	// not require the initial position, but we need to set
	// it anyway
	Mat_<float> measurement(2,1);
	measurement.setTo(Scalar(0));

	// We will detect the mouse position inside this window.
	// We set the function on_mouse on the newly created window.
	namedWindow("Kalman");
	setMouseCallback("Kalman", on_mouse, 0);

	// Set up the initial state of the Kalman filter
	// (we do not set the velocity nor the acceleration).
	KF.statePre.at<float>(0) = mouse_info.x;
	KF.statePre.at<float>(1) = mouse_info.y;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;
	KF.statePre.at<float>(4) = 0;
	KF.statePre.at<float>(5) = 0;

	// We need then to define the transition matrix.
	// It has to be 4x4 in order to get the following equations:
	// -> x = x_0 + t*v_x + 1/2*a_x*t^2
	// -> y = y_0 + t*v_y + 1/2*a_y*t^2
	// -> v_xt+1 = v_tx+a_xt
	// -> v_ut+1 = v_ty+a_yt
	// -> a_xt+1 = a_xt
	// -> a_yt+1 = a_yt+1
	// Therefore, T must be:
	// 1 0 1 0 1/2 0
	// 0 1 0 1 0 1/2
	// 0 0 1 0 0 0
	// 0 0 0 1 0 0
	// 0 0 0 0 1 0
	// 0 0 0 0 0 1
	// such that T*S will give us those equations.
	KF.transitionMatrix = *(Mat_<float>(6,6) <<
	1,0,1,0,0.5,0,
	0,1,0,1,0,0.5,
	0,0,1,0,1,0,
	0,0,0,1,0,1,
	0,0,0,0,1,0,
	0,0,0,0,0,1);

	// Set the measurement function to the identity
	// and to the process noise covariance. The covariance
	// will have slightly smaller values on the diagonal.
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(0.1));
	setIdentity(KF.errorCovPost, Scalar::all(0.1));

	// Clear the vectors
	mousev.clear();
	kalmanv.clear();

	// Update process
	while(true) {

		// Predict. Updates the statePre variable.
		// We store also its value to plot it later
		// (basically its x and y coordinates).
		Mat prediction = KF.predict();
		Point predictionPt(prediction.at<float>(0), prediction.at<float>(1));

		// Measurement
		measurement(0) = mouse_info.x;
		measurement(1) = mouse_info.y;

		// Store the mouse position to plot it.
		Point measPt(measurement(0), measurement(1));
		mousev.push_back(measPt);

		// Update phase
		Mat estimated = KF.correct(measurement);

		// Store the kalman predicted point
		Point statePt(estimated.at<float>(0), estimated.at<float>(1));
		kalmanv.push_back(statePt);

		img = Scalar::all(0);

		// State point plotted in white
		// Measured point will be plotted in red
		// Predicted point will be plotted in green
		drawCross(statePt, Scalar(255, 255, 255), 5);
		drawCross(measPt, Scalar(0,0,255), 5);
		drawCross(predictionPt, Scalar(0,255,0), 5);

		// Print a line over all the points
		for (int i = 0; i < mousev.size()-1; ++i) {
			line(img, mousev[i], mousev[i+1], Scalar(255,0,0));
		}

		// Printe the same for the kalman points
		// (prediction of the kalman filter)
		for (int j = 0; j < kalmanv.size()-1; ++j) {
			line(img, kalmanv[j], kalmanv[j+1], Scalar(0,255,0));
		}

		// Show the image
		imshow("Kalman", img);
		waitKey(1);

	}

	return 0;
}

