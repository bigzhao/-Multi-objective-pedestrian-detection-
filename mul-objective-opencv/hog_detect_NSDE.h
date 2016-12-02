

class HOG_NSDE:public HOGDescriptor
{
public:

	void compute_2(const Mat& img, vector<float>& descriptors,Size winStride, Size padding,const vector<Point>& locations);
	bool setImage_2(const Mat& image);

};