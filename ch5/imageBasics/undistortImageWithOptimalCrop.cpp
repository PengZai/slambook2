#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";   // 请确保路径正确


class DistortModel
{
  public:

  DistortModel(int new_w, int new_h, int wOrg, int hOrg);
  virtual ~DistortModel(){};

  virtual void distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
 float* out_u, float* out_v) = 0;

  void makeOptimalKCrop();
  void undistort(cv::Mat& undistorted_image, float* image_raw, int wOrg, int hOrg);


  protected:
  int wOrg_;
  int hOrg_;
  int new_w_;
  int new_h_;
  float new_fx_ = 1.0;
  float new_fy_ = 1.0;
  float new_cx_ = 0.0;
  float new_cy_ = 0.0;

  float* remapU_ = nullptr;
  float* remapV_ = nullptr;

};

DistortModel::DistortModel(int new_w, int new_h, int wOrg, int hOrg):
new_w_(new_w),
new_h_(new_h),
wOrg_(wOrg),
hOrg_(hOrg)
{

  remapU_ = new float[new_w*new_h]();
  remapV_ = new float[new_w*new_h]();

  

}


void DistortModel::makeOptimalKCrop()
{


  for(int v=0;v<new_h_;v++)
  {
	  for(int u=0;u<new_w_;u++)
		{
      int coords = v*new_w_+u;
			remapU_[coords] = u;
			remapV_[coords] = v;
		}
  }
  
  new_fx_ = 1.0;
  new_fy_ = 1.0;
  new_cx_ = 0.0;
  new_cy_ = 0.0;

  float* tgX = new float[100000]();
	float* tgY = new float[100000]();
	float minX = 0;
	float maxX = 0;
	float minY = 0;
	float maxY = 0;

  // convert into normal space between [-5,5]
	for(int x=0; x<100000;x++)
  {
    tgX[x] = (x-50000.0f) / 10000.0f; 
    tgY[x] = 0;
  }
  distortCoordinates(tgX, tgY, 100000, new_fx_, new_fy_, new_cx_, new_cy_, tgX, tgY);
  for(int x=0; x<100000;x++)
	{
		if(tgX[x] > 0 && tgX[x] < wOrg_-1)
		{
			if(minX==0) minX = (x-50000.0f) / 10000.0f;
			maxX = (x-50000.0f) / 10000.0f;
		}
	}
  for(int y=0; y<100000;y++)
	{
    tgY[y] = (y-50000.0f) / 10000.0f;
    tgX[y] = 0;
  }
  distortCoordinates(tgX, tgY, 100000, new_fx_, new_fy_, new_cx_, new_cy_, tgX, tgY);
	for(int y=0; y<100000;y++)
	{
		if(tgY[y] > 0 && tgY[y] < hOrg_-1)
		{
			if(minY==0) minY = (y-50000.0f) / 10000.0f;
			maxY = (y-50000.0f) / 10000.0f;
		}
	}

  delete[] tgX;
	delete[] tgY;

	minX *= 1.01;
	maxX *= 1.01;
	minY *= 1.01;
	maxY *= 1.01;

	printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);


  bool hasBlackLeft = true, hasBlackRight = true, hasBlackTop = true, hasBlackBottom = true;
  int iteration_count = 0;
  while(hasBlackLeft || hasBlackRight || hasBlackBottom || hasBlackTop)
  {
    hasBlackLeft = hasBlackRight = hasBlackTop = hasBlackBottom = false;
    for(int v=0;v<new_h_;v++)
    {
      remapU_[v] = minX;
      remapU_[v+new_h_] = maxX;
      remapV_[v] = remapV_[v+new_h_] = minY + (maxY-minY)*(float)v/(new_h_-1.0f);
    }
    distortCoordinates(remapU_, remapV_, 2*new_h_, new_fx_, new_fy_, new_cx_, new_cy_, remapU_, remapV_);

    for(int v=0;v<new_h_;v++)
    {
  
        if(remapU_[v] <= 0 || remapU_[v] >= wOrg_-1){
          hasBlackLeft = true;
        }
        if(remapU_[v+new_h_] <= 0 || remapU_[v+new_h_] >= wOrg_-1){
          hasBlackRight = true;
        }

    }


    for(int u=0;u<new_w_;u++)
    {
      remapV_[u] = minY;
      remapV_[u+new_w_] = maxY;
      remapU_[u] = remapU_[u+new_w_] = minX + (maxX-minX)*(float)u/(new_w_-1.0f);
    }
    distortCoordinates(remapU_, remapV_, 2*new_w_, new_fx_, new_fy_, new_cx_, new_cy_, remapU_, remapV_);

    for(int u=0;u<new_w_;u++)
    {
  
        if(remapV_[u] <= 0 || remapV_[u] >= hOrg_-1){
          hasBlackTop = true;
        }
        if(remapV_[u+new_w_] <= 0 || remapV_[u+new_w_] >= hOrg_-1){
          hasBlackBottom = true;
        }

    }

    if((hasBlackLeft || hasBlackRight) && (hasBlackTop || hasBlackBottom))
    {
      if((maxX-minX) > (maxY-minY))
          hasBlackBottom = hasBlackTop = false;	// only shrink left/right
      else
          hasBlackLeft = hasBlackRight = false; // only shrink top/bottom
    }

    if(hasBlackLeft) minX *= 0.995;
		if(hasBlackRight) maxX *= 0.995;
		if(hasBlackTop) minY *= 0.995;
		if(hasBlackBottom) maxY *= 0.995;

		iteration_count++;


    printf("iteration %05d: range: x: %.4f - %.4f; y: %.4f - %.4f!\n", iteration_count,  minX, maxX, minY, maxY);
		if(iteration_count > 500)
		{
			printf("FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING \n");
			std::exit(1);
		}


  } 


	new_fx_ = ((float)new_w_)/(maxX-minX);
	new_fy_ = ((float)new_h_)/(maxY-minY);
	new_cx_ = -minX*new_fx_;
	new_cy_ = -minY*new_fy_;

  printf("new_fx: %.4f, new_fy: %.4f, new_cx: %.4f, new_cy: %.4f\n", new_fx_, new_fy_, new_cx_, new_cy_);

  for(int v=0;v<new_h_;v++)
  {
    for(int u=0;u<new_w_;u++)
    {
      remapU_[u+v*new_w_] = u;
      remapV_[u+v*new_w_] = v;
    }

  }


  distortCoordinates(remapU_, remapV_, new_w_*new_h_, new_fx_, new_fy_, new_cx_, new_cy_, remapU_, remapV_);

}




void DistortModel::undistort(cv::Mat& undistorted_image, float* image_raw, int wOrg, int hOrg)
{

  for(int v=0;v<new_h_;v++)
  {
	  for(int u=0;u<new_w_;u++)
		{
      int coordinate = v*new_w_+u;
			float u_distorted = remapU_[coordinate];
			float v_distorted = remapV_[coordinate];

      if (u_distorted >= 0 && v_distorted >= 0 &&
                u_distorted < wOrg - 1 && v_distorted < hOrg - 1)
        {
            int u0 = (int)u_distorted;
            int v0 = (int)v_distorted;
            float du = u_distorted - u0;
            float dv = v_distorted - v0;

            float I00 = image_raw[v0 * wOrg + u0];
            float I10 = image_raw[v0 * wOrg + (u0 + 1)];
            float I01 = image_raw[(v0 + 1) * wOrg + u0];
            float I11 = image_raw[(v0 + 1) * wOrg + (u0 + 1)];

            float val =
                (1 - du) * (1 - dv) * I00 +
                du * (1 - dv) * I10 +
                (1 - du) * dv * I01 +
                du * dv * I11;

            undistorted_image.at<uchar>(v, u) = val;
        }

      // if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < wOrg && v_distorted < hOrg) {
      //   undistorted_image.at<uchar>(v, u) = image_raw[(int)v_distorted*wOrg+(int)u_distorted];
      // }
      else
      {
          undistorted_image.at<uchar>(v, u) = 0;
          std::cout << "wrong undistort coordinate v:" << v << ",u:" << u << std::endl;
      }
		}
  }

}


class EquidistantDistortModel : public DistortModel
{

  public:

  EquidistantDistortModel(int new_w, int new_h, int wOrg, int hOrg);

  void distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
 float* out_u, float* out_v);

  protected:
  // 畸变参数
  float k1 = -0.04345139283609733;
  float k2 = 0.019439878275353862;
  float k3 = -0.03544505860721041;
  float k4 = 0.022121647569599227;

  // 内参
  float fx = 1059.6087870662222;
  float fy = 1059.5973705610731;
  float cx = 1050.3552896004385;
  float cy = 731.3912426749002;


};


EquidistantDistortModel::EquidistantDistortModel(int new_w, int new_h, int wOrg, int hOrg)
:DistortModel(new_w, new_h, wOrg, hOrg){}



void EquidistantDistortModel::distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
 float* out_u, float* out_v)
{


    for (int i = 0; i < n; i++) {
          // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
          float x = (in_u[i] - in_cx) / in_fx;
          float y = (in_v[i] - in_cy) / in_fy;

          float r_u = std::sqrt(x * x + y * y);


          // Compute theta (angle from optical axis)
          float theta = std::atan(r_u);
          float theta2 = theta * theta;
          float theta4 = theta2 * theta2;
          float theta6 = theta4 * theta2;
          float theta8 = theta4 * theta4;
          // Distorted angle
          float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

          // Mapping back to distorted normalized coordinates
          // double scale = (theta_d / r_u);
          float scale = (r_u > 1e-8) ? thetad / r_u : 1.0;

          float x_d = x * scale;
          float y_d = y * scale;

          // Convert to pixel coordinates
          float u_distorted = fx * x_d + cx;
          float v_distorted = fy * y_d + cy;

          out_u[i] = u_distorted;
          out_v[i] = v_distorted;
        
    }
}



class RadtanDistortModel : public DistortModel
{

  public:

  RadtanDistortModel(int new_w, int new_h, int wOrg, int hOrg);

  void distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
 float* out_u, float* out_v);

  protected:


  // 畸变参数
  // float k1 = -0.28340811;
  // float k2 = 0.07395907;
  // float p1 = 0.00019359;
  // float p2 = 1.76187114e-05;


  // // 内参
  // float fx = 458.654;
  // float fy = 457.296;
  // float cx = 367.215;
  // float cy = 248.375;

  // 畸变参数
  float k1 = -0.060164620903866;
  float k2 = 0.094005180631043;
  float p1 = 0;
  float p2 = 0;

  // 内参
  float fx = 642.9165664800531;
  float fy = 641.9171825800378;
  float cx = 460.1840658156501;
  float cy = 308.5846449100310;


};


RadtanDistortModel::RadtanDistortModel(int new_w, int new_h, int wOrg, int hOrg)
:DistortModel(new_w, new_h, wOrg, hOrg){}



void RadtanDistortModel::distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
 float* out_u, float* out_v)
{


    for (int i = 0; i < n; i++) {
          // 按照公式，计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)

          float x = (in_u[i] - in_cx) / in_fx;
          float y = (in_v[i] - in_cy) / in_fy;
          float r = sqrt(x * x + y * y);
          float x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
          float y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
          float u_distorted = fx * x_distorted + cx;
          float v_distorted = fy * y_distorted + cy;

          out_u[i] = u_distorted;
          out_v[i] = v_distorted;
        
    }
}


int main(int argc, char **argv) {


  // int new_h = 480;
  // int new_w = 640;
  // int new_h = 480;
  // int new_w = 752;
  int new_h = 600;
  int new_w = 960;

  // int new_h = 1536;
  // int new_w = 2048;

  cv::Mat image = cv::imread(argv[1], 0);   // 图像是灰度图，CV_8UC1
  cv::Mat undistorted_image = cv::Mat(new_h, new_w, CV_8UC1);

  int wOrg = image.cols;
  int hOrg = image.rows;
  float* image_raw = new float[wOrg*hOrg]();

  for(int v=0;v<hOrg;v++){
    for(int u=0;u<wOrg;u++){
      image_raw[v*wOrg+u] = image.at<uchar>(v,u);
    }
  }

  // DistortModel* distortmodel =  new EquidistantDistortModel(new_w, new_h, wOrg, hOrg);
  DistortModel* distortmodel =  new RadtanDistortModel(new_w, new_h, wOrg, hOrg);

  distortmodel->makeOptimalKCrop();

  distortmodel->undistort(undistorted_image, image_raw, wOrg, hOrg);


  // 画图去畸变后图像
  cv::imwrite("undistorted.png", undistorted_image);
  cv::imshow("distorted", image);
  cv::imshow("undistorted", undistorted_image);
  cv::waitKey();
  return 0;
}
