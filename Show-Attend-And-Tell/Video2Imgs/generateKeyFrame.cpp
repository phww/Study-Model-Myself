// @Time : 2021/5/29 下午8:20
// @Author : PH
// @Version：V 0.1
// @File : generateKeyFrame.cpp
// @desc :
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <boost/algorithm/string.hpp>
#include "getFiles.h"
using namespace std;
using namespace cv;
vector<Mat> getHistRGB(Mat& frame) {
  vector<Mat> hist_rgb(3);
  // 分离BGR通道
  Mat channels[3];
  split(frame, channels);
  // 分别计算每个通道的直方图
  const int bins[1] = {256};
  const float inranges[2] = {0, 255};
  const float* ranges = {inranges};
  calcHist(&channels[0], 1, 0, Mat(), hist_rgb[0], 1, bins, &ranges);
  calcHist(&channels[1], 1, 0, Mat(), hist_rgb[1], 1, bins, &ranges);
  calcHist(&channels[2], 1, 0, Mat(), hist_rgb[2], 1, bins, &ranges);
  return hist_rgb;
}

double autoThreshold(vector<double>& nums) {
  double mean = std::accumulate(nums.begin(), nums.end(), 0) / nums.size();
  double var = 0;
  for (auto num : nums) {
	var += pow(num - mean, 2);
  }
  var /= nums.size();
  double std = sqrt(var);
  return std + mean;
}

int main() {
  vector<string> video_paths;
  getFiles("/home/ph/Dataset/VideoCaption/test", video_paths);
  cout << "待处理视频数：" << video_paths.size() << endl;
  for (int i = 0; i < video_paths.size(); i++) {
	cout << "处理视频：" << i + 1 << " ";
	// 捕获一个视频
	string video_path = video_paths[i];
	vector<string> split_str;
	boost::split(split_str, video_path, boost::is_any_of("./"));
	string video_name = split_str[split_str.size() - 2];
	VideoCapture cap;
	cap.open(video_path);
	if (!cap.isOpened()) {
	  cout << "can not open video!" << endl;
	}

	// 视频信息
	int n_frame = cap.get(CAP_PROP_FRAME_COUNT);

	// 三帧直方图差分
	Mat frame1, frame2, cur_frame;
	vector<double> all_scores;
	for (int j = 0; j <= n_frame; j++) {
	  // 先获取前两帧
	  if (j == 0) {
		cap >> frame1;
		string name = "/home/ph/Dataset/VideoCaption/generateImgs/test/" + video_name + "-" + to_string(j) + ".jpg";
		imwrite(name, frame1);
		cout << "保存第" << j << "帧" << " ";
		continue;
	  }
	  if (j == 1) {
		cap >> frame2;
		continue;
	  }

	  // 从第2帧开始计算直方图差分的分数
	  cap >> cur_frame;
	  if (cur_frame.empty()) {
		break;
	  }
	  vector<Mat> A, B, C;
	  A = getHistRGB(frame1);
	  B = getHistRGB(frame2);
	  C = getHistRGB(cur_frame);

	  double score1 = 0, score2 = 0;
	  for (int k = 0; k < 3; k++) {
		score1 += compareHist(A[k], B[k], HISTCMP_INTERSECT);
		score2 += compareHist(B[k], C[k], HISTCMP_INTERSECT);
	  }
	  all_scores.push_back(abs(score2 - score1)/3);

	  // 以三帧为窗口，滑动窗口计算整个视频的差分结果
	  frame2.copyTo(frame1);
	  cur_frame.copyTo(frame2);
	}

	// 从视频的第二帧开始，根据全部差分结果和自适应阈值来确定关键帧
	double threshold = autoThreshold(all_scores);
	Mat key_frame;
	int cnt = 0;
	for (int k = 2; k <= n_frame; k++) {
	  if (all_scores[k - 1] > threshold) {
		cap.set(CAP_PROP_POS_FRAMES, k);
		cap >> key_frame;
		k++; // 一般来说第k帧和第k+1帧差不多，只要一个就行了
		cnt++; // 对一个视频最多只要8帧
		if (key_frame.empty() || cnt > 8) {
		  break;
		}
		string name = "/home/ph/Dataset/VideoCaption/generateImgs/test/" + video_name + "-" + to_string(k) + ".jpg";
		imwrite(name, key_frame);
		cout << "保存第" << k << "帧" << " ";
	  }
	}
	cout << endl;
	cap.release();
  }
  return 0;
}

