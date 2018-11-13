/*
* Written (W) 2018 uriel
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

//Define path to pre-trained classifier
String face_cascade_name = "../data/haarcascade_frontalface_alt.xml";

CascadeClassifier face_cascade;

// ViolaJones algorithm
int main(int argc, char ** argv)
{
	VideoCapture cap(0);
	Mat frame;

	// Try to load the pre-trained model
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "Error loading the model" << endl;
		return -1;
	}

	// Read the video stream
	while(true)
	{
		cap >> frame;

		// Store the results
		vector<Rect> faces;
		Mat frame_gray; // The classfier works only on gray frames

		// Convert the image
		cvtColor(frame, frame_gray, CV_BGR2GRAY);

		// Normalize the image (equalize the histogram)
		equalizeHist(frame_gray, frame_gray);

		// Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));

		// Plot the faces into the image to check if they are correct.
		// Place an ellipsis for each detected face.
		for (int i=0; i< faces.size(); i++)
		{
			// Find the center of the face
			Point center (faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);

			// Plot the ellipse
			ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0,0,360, Scalar(0,0,255), 4, 8,0);

		}

		imshow("Result", frame);
		waitKey(1);

	}


	return 0;
}
