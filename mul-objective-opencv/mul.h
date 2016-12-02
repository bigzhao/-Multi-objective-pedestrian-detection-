#ifndef __MUL__
#define __MUL__
#include "opencv2/opencv.hpp" 


#include "cv.h" 
#include "cv.hpp" 
#include "cxcore.h" 
#include "cxcore.hpp" 

#include<cmath>

#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetec.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "highgui.h" 
//#include <opencv2\objdetect\objdetect_c.h> 

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <vector>
#include <algorithm>
#define NP 200
#define PI 3.1415926
#define Dim 3
#define NICHING_SIZE 20
#define LOCAL_DELTA  0.001
#define LOCAL_SEARCHING_TIMES 4
#define X2_SCALE  10.0
#define THRESHOLD 2.5
boost::mt19937 rng((unsigned)time(0)); /* 随机数种子 */
boost::uniform_01<boost::mt19937&> u01(rng);
using namespace cv;
using namespace std;
#endif 