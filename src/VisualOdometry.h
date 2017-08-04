/*
 * VisualOdometry.h
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#ifndef VISUALODOMETRY_H_
#define VISUALODOMETRY_H_

#include "TSDFModel.h"
#include "ICPOdometry.h"
#include "Frame.h"
#include "Camera.h"
#include "common_include.h"
#include <iomanip>
#include <fstream>
#include <chrono>

namespace myfusion {

class VisualOdometry {
public:
    typedef std::shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        INITIALIZING=-1,
        OK=0,
        LOST
    };

    VOState     state;     // current VO status

    ICPOdometry::Ptr icp_vo;
    TSDFModel::Ptr tsdf_build;

    Frame::Ptr  ref;       // reference key-frame
    Frame::Ptr  curr;      // current frame


    SE3 T_c_w_estimated_;    // the estimated pose of current frame
    int num_inliers_;        // number of inlier features in icp
    int num_lost_;           // number of lost times

	float dist_thresh;
	float angle_thresh;

	const int width;
	const int height;
	const float cx, cy, fx, fy;

	int threads;
    int blocks;

public: // functions
    VisualOdometry(int width, int height, int threads, int blocks, Camera::Ptr& camera);
    ~VisualOdometry();

    void addFrame(Frame::Ptr frame);
    void outputModel();

};

} /* namespace myfusion */

#endif /* VISUALODOMETRY_H_ */
