/*
 * Camera.cpp
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#include "Camera.h"

namespace myfusion {

Camera::Camera() {
	// TODO Auto-generated constructor stub
	fx_ = Config::get<float>("camera.fx");
	fy_ = Config::get<float>("camera.fy");
	cx_ = Config::get<float>("camera.cx");
	cy_ = Config::get<float>("camera.cy");
	depth_scale_ = Config::get<float>("camera.depth_scale");
}



} /* namespace myfusion */
