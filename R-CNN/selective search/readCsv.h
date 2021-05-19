// @Time : 2021/5/12 下午1:46
// @Author : PH
// @Version：V 0.1
// @File : readCsv.t
// @desc : 读取Airplane数据集中的标注，并将csv格式的标注转换为CV::Rect格式
#ifndef _READCSV_H
#define _READCSV_H
#include <boost/algorithm/string.hpp> //使用里面的spilt分割字符串
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

/*!
 * @brief 读取csv格式的标注，并保存到cv::Rect对象中去
 * @param csv_path csv文件路径
 * @param rects 保存cv::Rect的vector
 */
void readCsv(const string &csv_path, vector<cv::Rect> &rects) {

  ifstream fin;
  fin.open(csv_path); // 打开csv文件到文件流fin
  string line;
  getline(fin, line); // 第一行是该csv文件内有几个标注，不需要使用这个信息
  // 每次读取文件流中的一行，保存在line中
  while (getline(fin, line)) {
	vector<string> line_spilt;
	cv::Rect rect;
	// 用boost库内的spilt分割字符串。即将"1 2 3 4"分割为"1","2","3","4"
	boost::split(line_spilt, line, boost::is_any_of(" "));
	// Airplane数据集内的标注是以左上角坐标(x1，y1)和右下角坐标(x2,y2)的形式表示矩形框的
	// 而cv::Rect 是以左上角坐标(x,y)和矩形的width和height表示矩形框
	int x1 = stoi(line_spilt[0]), y1 = stoi(line_spilt[1]);
	int x2 = stoi(line_spilt[2]), y2 = stoi(line_spilt[3]);
	rect.x = x1, rect.y = y1;
	rect.width = abs(x1 - x2), rect.height = abs(y1 - y2);
	rects.push_back(rect);
  }
}
#endif
//int main() {
//  string csv_path = "./image/test.csv";
//  string img_path = "./image/test.jpg";
//  vector<cv::Rect> rects;
//  readCsv(csv_path, rects);
//  cv::Mat img = cv::imread(img_path);
//  for (const auto &rect : rects) {
//	cout << rect.x << " " << rect.y << " " << rect.height << " " << rect.width << endl;
//	cv::rectangle(img, rect, cv::Scalar(255, 0, 0));
//  }
//  cv::namedWindow("img", cv::WINDOW_FREERATIO);
//  cv::imshow("img", img);
//  cv::waitKey(0);
//  cv::destroyAllWindows();
//  return 0;
//}

