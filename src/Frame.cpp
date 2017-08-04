/*
 * Frame.cpp
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#include "Frame.h"

namespace myfusion {

Frame::Frame() : id(-1), time_stamp(-1), camera(nullptr), is_key_frame(false),
		width_depth(0), height_depth(0), width_texture(0), height_texture(0)
{
	// TODO Auto-generated constructor stub
}

void Frame::CreateNewFrame(Camera::Ptr &camera, Mat& depth, Mat& color)
{
	camera = camera;
	width_depth = depth.cols;
	height_depth = depth.rows;

	width_texture = color.cols;
	height_texture = color.rows;

	cout << "color channel" << color.channels() << endl;
	cout << "depth channel" << depth.channels() << endl;

	depth_data = new unsigned char[depth.channels() * width_depth * height_depth * 2];
	texture_data = new unsigned char[color.channels() * width_texture * height_texture];

	memcpy(depth_data, depth.data, depth.channels() * width_depth * height_depth * 2);
	memcpy(texture_data, color.data, color.channels() * width_texture * height_texture);

	unsigned short* depth_run = (unsigned short*)depth_data;
	for(unsigned int i = 0; i < height_depth; i++)
	{
		for(unsigned int j = 0; j < width_depth; j++)
		{
			*depth_run = (*depth_run)/5;
			depth_run++;
		}
	}

}

Frame::~Frame() {
	cout << "frame release" << endl;
	if(depth_data)
	{
		delete[] depth_data;
		depth_data = nullptr;
	}
	if(texture_data)
	{
		delete[] texture_data;
		texture_data = nullptr;
	}
}

} /* namespace myfusion */
