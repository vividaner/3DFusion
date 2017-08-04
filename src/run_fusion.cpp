// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Config.h"
#include "VisualOdometry.h"
#include "common_include.h"
#include "TSDFModel.h"




void tokenize(const std::string & str, std::vector<std::string> & tokens, std::string delimiters = " ")
{
    tokens.clear();

    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    std::string::size_type pos = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = str.find_first_not_of(delimiters, pos);
        pos = str.find_first_of(delimiters, lastPos);
    }
}

int main ( int argc, char** argv )
{
    if ( argc < 3 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

	myfusion::Config::setParameterFile ( argv[1] );
	myfusion::Camera::Ptr camera(new myfusion::Camera);
	myfusion::VisualOdometry::Ptr vo(new myfusion::VisualOdometry(640, 480, 224, 96, camera));

	string dataset_dir = myfusion::Config::get < string > ("dataset_dir");
	cout << "dataset: " << dataset_dir << endl;
	string rgb_file, depth_file;
	unsigned int frame_num_int = atoi(argv[2]);
	std::fstream depth_list, rgb_list;
	depth_list.open(argv[3]);
	rgb_list.open(argv[4]);
	if(!depth_list.is_open() || !rgb_list.is_open())
	{
		cout << "file open error" << endl;
		exit(1);
	}




	// visualization
	cv::viz::Viz3d vis("Visual Odometry");
	cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
	cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0,
			1, 0);
	cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point,
			cam_y_dir);
	vis.setViewerPose(cam_pose);

	world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
	camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
	vis.showWidget("World", world_coor);
	vis.showWidget("Camera", camera_coor);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::string dev(prop.name);
	std::cout << dev << std::endl;

	myfusion::Config::setParameterFile(argv[1]);
	cout << "read total " << frame_num_int << " entries" << endl;
	for (int i = 0; i < frame_num_int; i++) {
		cout << "****** loop " << i << " ******" << endl;
#ifdef DATA
		if (i < 10) {
			rgb_file = dataset_dir + "img/" + "0000" + to_string(i) + ".jpg";
			depth_file = dataset_dir + "depth/" + "0000" + to_string(i)
					+ ".png";
		} else if (i < 100) {
			rgb_file = dataset_dir + "img/" + "000" + to_string(i) + ".jpg";
			depth_file = dataset_dir + "depth/" + "000" + to_string(i) + ".png";
		} else if (i < 1000) {
			rgb_file = dataset_dir + "img/" + "00" + to_string(i) + ".jpg";
			depth_file = dataset_dir + "depth/" + "00" + to_string(i) + ".png";
		} else {
			rgb_file = dataset_dir + "img/" + "0" + to_string(i) + ".jpg";
			depth_file = dataset_dir + "depth/" + "0" + to_string(i) + ".png";
		}
#else
		string curr_line_depth, curr_line_rgb;

		vector<string> tokens_depth, tokens_rgb;
		do {
			getline(depth_list, curr_line_depth);
			tokenize(curr_line_depth, tokens_depth);
		} while (tokens_depth.size() > 2);

		do {
			getline(rgb_list, curr_line_rgb);
			tokenize(curr_line_rgb, tokens_rgb);
		} while(tokens_rgb.size() > 2);

		depth_file = dataset_dir + tokens_depth[1];
		rgb_file = dataset_dir + tokens_rgb[1];
#endif

		//cout << rgb_file << endl;
		//cout << depth_file << endl;

		Mat color = cv::imread(rgb_file);
		Mat depth = cv::imread(depth_file, cv::IMREAD_ANYDEPTH);
		if (color.data == nullptr || depth.data == nullptr) {
			cout << "file read error\n" << endl;
			break;
		}
		myfusion::Frame::Ptr pFrame(new myfusion::Frame);
		pFrame->CreateNewFrame(camera, depth, color);

		vo->addFrame(pFrame);

	}

	vo->outputModel();

    cv::waitKey();

    return 0;
}
