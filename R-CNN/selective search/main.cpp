#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <boost/algorithm/string.hpp>
#include "getFiles.h"
#include "readCsv.h"
#include <ctime>
using namespace std;

/*!
 * 计算两个rect对象的iou
 * @param rects： selected search算法得到的rect
 * @param gt_rect：标注的ground truth
 * @return iou：两个rect的交并比
 */
double calcIOU(const cv::Rect &rects, const cv::Rect &gt_rect) {
  // 使用cv::Rect表示bbox的优势就是：可以方便的使用a|b表示并集，a&b表示交集
  cv::Rect I = rects & gt_rect;
  cv::Rect U = rects | gt_rect;
  double iou = I.area() * 1.0 / U.area();
  return iou;
}

int main() {
  clock_t start_time = clock();
  // 设置自动优化和多线程，可以加速程序
  cv::setUseOptimized(true);
  cv::setNumThreads(8);
  // select_search对象
  cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation>
	  ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

  // 读取image目录下的所有图片
  vector<string> gt_paths;
  vector<string> img_paths;
  getFiles("/home/ph/Dataset/Airplane/Airplanes_Annotations", gt_paths);
  getFiles("/home/ph/Dataset/Airplane/Images", img_paths);
//  getFiles("./image/gt", gt_paths);
//  getFiles("./image/img", img_paths);
  // sort竟然支持string数组的排序！！！
  // 对img_paths和gt_paths排序，保证两者同一下标表示的图片和标注能够匹配
  sort(gt_paths.begin(), gt_paths.end());
  sort(img_paths.begin(), img_paths.end());
  cout << "待处理图片总数量：" << img_paths.size() << endl;

  // 下面用selected search逐个处理图片
  for (int i = 0; i < img_paths.size(); i++) {
	string img_path = img_paths[i];
	string gt_path = gt_paths[i];
	// cv::Rect 内包含4个元素，矩形的左上角坐标(x，y)以及矩形的宽w和长h
	vector<cv::Rect> rects;// 保存region的vector,其中selected search算法返回的是opencv中的Rect类型的数据
	// 读取图片和ground_truth
	cv::Mat img = cv::imread(img_path, 1);
	vector<cv::Rect> gt_rects;
	readCsv(gt_path, gt_rects);
	// 使用switch...自动选择处理strategy时，需要先设置一个base_img。然后才能给出对应的strategy
	ss->setBaseImage(img); //设置为base_img
	// 设置处理精度
	ss->switchToSelectiveSearchQuality();
	// 处理图像,结果保存在rects中
	ss->process(rects);
	cout << "处理图片：" << i + 1 << " " << "候选区域数量：" << rects.size() << endl;

	// 处理这些regions
	int t_cnt = 0, f_cnt = 0; //统计正列和负例的数量，一般一副图片生成的regions中只要30个正列、负例就行
	vector<string> split_path;
	//获取图片的名称，方便下面给新图片命名使用
	boost::split(split_path, img_path, boost::is_any_of("/")); //c++的STL连个split都没有！！？
	string img_name = split_path.back();
	//当前处理图片，由selected search算法得到的每个region和标注的gt_rect都要计算iou
	for (int j = 0; j < rects.size() && j < 2000; j++) {
	  cv::Rect rect = rects[j];
	  int cnt = 0; //下面判断负例依据
	  for (const auto &gt_rect : gt_rects) {
		// 依次计算iou
		double iou = calcIOU(rect, gt_rect);
		// 根据iou为每个region打上标签
		// 如果这个region与标注中的任意一个gt_rect的iou>0.7就判断其为正例
		// 如果这个region与标注中的所有gt_rect的iou都小于0.3,就判断其为负例。因此要用cnt统计一下该信息
		if (iou > 0.7 && t_cnt <= 30) {
		  cv::Mat region(img, rect); // Mat可以使用原图像加rect的方式切割原图片的形式初始化
		  //resize到统一大小，方便之后输入到CNN中基
		  //这里不使用默认的双线性差值的方法，使用基于像素面积的采样方法
		  cv::resize(region, region, cv::Size(224, 224), cv::INTER_AREA);
		  //使用字符流来实现格式化字符串，为新图像命名，命名规则：j+正例(1)负例(2)+图像原名
		  ostringstream file_name;
		  file_name << "./region/" << j << "-1-" << img_name;
		  cv::imwrite(file_name.str(), region);
		  t_cnt++;
		} else if (iou < 0.3) {
		  cnt++;
		}
	  }
	  // 当cnt等于当前处理图片标注的数量时，代表该region是个背景（负例）
	  if (cnt == gt_rects.size() && f_cnt <= 30) {
		cv::Mat region(img, rect);
		cv::resize(region, region, cv::Size(224, 224), cv::INTER_AREA);
		ostringstream file_name;
		file_name << "./region/" << j << "-0-" << img_name;
		cv::imwrite(file_name.str(), region);
		f_cnt++;
	  }
	}
  }
  clock_t end_time = clock();
  cout<<"Done!"<<" "<<"total_time"<<(end_time - start_time)/CLOCKS_PER_SEC<<"s"<<endl;//2238s
  return 0;
}
