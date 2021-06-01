// @Time : 2021/5/11 下午1:15
// @Author : PH
// @Version：V 0.1
// @File : getFiles.h
// @desc : linux下递归获取文件夹下所有文件
#ifndef _GETFILES_H
#define _GETFILES_H
#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <cstring>

//struct dirent
//{
//  __ino_t d_ino;   /*inode number 索引节点号*/
//  __off_t d_off;   /*offset to this dirent 在目录文件中的偏移*/
//  unsigned short int d_reclen;   /*length of this d_name 文件名长*/
//  unsigned char d_type;   /*the type of d_name 文件类型*/
//  char d_name[256];    /*file name(null-terminated) 文件名 */
//};

using namespace std;
void getFiles(const string &root, vector<string> &files) {
  DIR *pDir; //指向根目录结构体的指针
  struct dirent *ptr; //dirent结构体指针，具体结构看开头的注释
  // 使用dirent.h下的opendir()打开根目录，并返回指针
  if (!(pDir = opendir(root.c_str()))) {
	return;
  }
  // 使用dirent.h下的readdir逐个读取root下的文件
  while ((ptr = readdir(pDir)) != nullptr) {
	// 这里我理解他的指针应该是自动会指向到下一个文件，所以不用写指针的移动
	string sub_file = root + "/" + ptr->d_name; // 当前指针指向的文件名
	if (ptr->d_type != 8 && ptr->d_type != 4) { // 递归出口，当不是普通文件（8）和文件夹（4）时退出递归
	  return;
	}
	// 普通文件直接加入到files
	if (ptr->d_type == 8) {
	  // 相当于将命令下使用ls展示出来的文件中除了. 和 ..全部保存在files中
	  // 当然下面可以写各种字符串的筛选逻辑，比如只要后缀有.jpg图片
	  if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
		if (strstr(ptr->d_name, ".mp4")) {
		files.push_back(sub_file);
		}
	  } // 当前文件为文件夹(4)类型，那就以当前文件为root进行递归吧！
	} else if (ptr->d_type == 4) {
	  // 同样下面也可以写文件夹名字的筛选逻辑，比如root/下有train，test文件夹，只遍历train文件夹内的文件
	  if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
		getFiles(sub_file, files);
	  }
	}
  }
  // 关闭根目录
  closedir(pDir);
}
#endif
//int main() {
//  string root = ".";
//  vector<string> files;
//  getFiles(root, files);
//  for (auto file : files) {
//	cout << file << endl;
//  }
//  return 0;
//}
