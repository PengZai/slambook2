#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./distorted.png";   // 请确保路径正确


void distortCoordinates(float* in_u, float* in_v, int n, float in_fx, float in_fy, float in_cx, float in_cy,
 float* out_u, float* out_v)
{

    // 畸变参数
    float k1 = -0.04345139283609733, k2 = 0.019439878275353862, k3 = -0.03544505860721041, k4 = 0.022121647569599227;
    // 内参
    float fx = 1059.6087870662222, fy = 1059.5973705610731, cx = 1050.3552896004385, cy = 731.3912426749002;


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

int main(int argc, char **argv) {


  float new_fx = 1.0, new_fy = 1.0, new_cx = 0.0, new_cy = 0.0;
  // int new_h = 480;
  // int new_w = 640;
  int new_h = 1536;
  int new_w = 2048;

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

  float* remapU = new float[new_w*new_h]();
  float* remapV = new float[new_w*new_h]();

  for(int v=0;v<new_h;v++)
  {
	  for(int u=0;u<new_w;u++)
		{
			remapU[v*new_w+u] = u;
			remapV[v*new_w+u] = v;
		}
  }
	
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
  distortCoordinates(tgX, tgY, 100000, new_fx, new_fy, new_cx, new_cy, tgX, tgY);
  for(int x=0; x<100000;x++)
	{
		if(tgX[x] > 0 && tgX[x] < wOrg-1)
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
  distortCoordinates(tgX, tgY, 100000, new_fx, new_fy, new_cx, new_cy, tgX, tgY);
	for(int y=0; y<100000;y++)
	{
		if(tgY[y] > 0 && tgY[y] < hOrg-1)
		{
			if(minY==0) minY = (y-50000.0f) / 10000.0f;
			maxY = (y-50000.0f) / 10000.0f;
		}
	}

  delete[] tgX;
	delete[] tgY;

	// minX *= 1.01;
	// maxX *= 1.01;
	// minY *= 1.01;
	// maxY *= 1.01;

	printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);

	new_fx = ((float)new_w-1.0f)/(maxX-minX);
	new_fy = ((float)new_h-1.0f)/(maxY-minY);
	new_cx = -minX*new_fx;
	new_cy = -minY*new_fy;
  
  distortCoordinates(remapU, remapV, new_w*new_h, new_fx, new_fy, new_cx, new_cy, remapU, remapV);


  for(int v=0;v<new_h;v++)
  {
	  for(int u=0;u<new_w;u++)
		{
			float u_distorted = remapU[v*new_w+u];
			float v_distorted = remapV[v*new_w+u];

      if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < wOrg && v_distorted < hOrg) {
        undistorted_image.at<uchar>(v, u) = image_raw[(int)v_distorted*wOrg+(int)u_distorted];
      } else {
        undistorted_image.at<uchar>(v, u) = 0;
      }
		}
  }

  

  // 画图去畸变后图像
  cv::imwrite("undistorted.png", undistorted_image);
  cv::imshow("distorted", image);
  cv::imshow("undistorted", undistorted_image);
  cv::waitKey();
  return 0;
}
