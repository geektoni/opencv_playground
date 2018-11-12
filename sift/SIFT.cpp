#include <stdio.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


/** @function main */
int main()
{

	// Read the images with 1 channel only (by putting
	// 0 as additional parameter)
  Mat img_object = imread("../data/box.png",0);
  Mat img_scene = imread("../data/box_in_scene.png",0);

  //cvtColor(img_object,img_object, CV_RGB2GRAY);
  //cvtColor(img_scene,img_scene, CV_RGB2GRAY);

  //-- Step 1: Detect the keypoints using SIFT Detector

  SiftFeatureDetector detector( 400 );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SiftDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  Mat keyPlot1, keyPlot2, keyPlot;

  cv::drawKeypoints(img_object,keypoints_object, keyPlot1,cv::Scalar(0,0,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  imshow("img1",keyPlot1);

  cv::drawKeypoints(img_scene,keypoints_scene, keyPlot2,cv::Scalar(0,0,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  imshow("img2",keyPlot2);

  //-- Step 3: Matching descriptor vectors using BF matcher
  BFMatcher matcher(NORM_L2);
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), 0 );

  imshow( "Matches", img_matches );

  //-- Draw only "good" matches
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 150 )
     { good_matches.push_back( matches[i]); }
  }

  Mat img_good_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_good_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), 0 );

  //-- Show detected matches
  imshow( "Good Matches", img_good_matches );

	// step 4 : stitching

	Mat result;

	// convert keypoint into sets of points2f
	vector<Point2f> obj, scene;

	// obj and scene now contains only keypoint corresponding to good match
	for (int j = 0; j < good_matches.size(); ++j) {
		obj.push_back(keypoints_object[good_matches[j].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[j].trainIdx].pt);
	}

	// Homography matrix (it maps the obj image to the scene image)
	Mat H = findHomography(obj, scene, CV_RANSAC);

	// Generate the resulting image from the Homography matrix
	warpPerspective(img_object, result, H, Size(img_scene.cols, img_scene.rows), INTER_CUBIC);

	imshow("result", result);

	// Generate the mask needed. The mask has still some
	// noise which could be removed with a dilation.
	Mat result_mask = Mat::zeros(result.size(), CV_8UC1);
	result_mask.setTo(255, result != 0);

	imshow("mask", result_mask);

	// Paste the resulting mask onto the original image
	result.copyTo(img_scene, result_mask);
	imshow("final_result", img_scene);

	waitKey(0);

  return 0;
  }

