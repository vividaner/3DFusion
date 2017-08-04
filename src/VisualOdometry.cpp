/*
 * VisualOdometry.cpp
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#include "VisualOdometry.h"

namespace myfusion {

uint64_t getCurrTime()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

void outputFreiburg(const std::string filename, const uint64_t & timestamp, const Eigen::Matrix4f & currentPose)
{
    std::ofstream file;
    file.open(filename.c_str(), std::fstream::app);

    std::stringstream strs;

    strs << std::setprecision(6) << std::fixed << (double)timestamp / 1000000.0 << " ";

    Eigen::Vector3f trans = currentPose.topRightCorner(3, 1);
    Eigen::Matrix3f rot = currentPose.topLeftCorner(3, 3);

    file << strs.str() << trans(0) << " " << trans(1) << " " << trans(2) << " ";

    Eigen::Quaternionf currentCameraRotation(rot);

    file << currentCameraRotation.x() << " " << currentCameraRotation.y() << " " << currentCameraRotation.z() << " " << currentCameraRotation.w() << "\n";

    file.close();
}

VisualOdometry::VisualOdometry(int width, int height, int threads, int blocks, Camera::Ptr& camera)
	: width (width), height(height), threads(threads), blocks(blocks),
	  cx(camera->cx_), cy(camera->cy_), fx(camera->fx_), fy(camera->fy_),
		icp_vo(new ICPOdometry(width, height, camera)), tsdf_build(new TSDFModel(width, height, camera))
{
	// TODO Auto-generated constructor stub
	state = INITIALIZING;
	tsdf_build->InitTSDFModel(camera);
}

void VisualOdometry::addFrame(Frame::Ptr frame)
{
	switch(state)
	{
	case INITIALIZING:
	{
		state = OK;
		curr = ref = frame;
		tsdf_build->UpdateData(frame);

		break;
	}
	case OK:
	{
		curr = frame;

		uint64_t time_start, time_end;
		time_start = getCurrTime();

		icp_vo->initICPModel((unsigned short*)ref->getDepthData());
		icp_vo->initICP((unsigned short*)curr->getDepthData());


		Sophus::SE3 T_prev_curr = curr->T_w_c;
		icp_vo->getIncrementalTransformation(T_prev_curr, threads, blocks);
		curr->T_w_c = ref->T_w_c * T_prev_curr;

		time_end = getCurrTime();
		cout << "ICP cost time " << (time_end - time_start)/1000.0f << endl;

		//cout << curr->T_w_c.cast<float>().matrix() << endl;
		//outputFreiburg("output.poses", curr->time_stamp, T_prev_curr.cast<float>().matrix());
		outputFreiburg("output.poses", curr->time_stamp, curr->T_w_c.cast<float>().matrix());

		time_start = getCurrTime();
		tsdf_build->UpdateData(frame);
		time_end = getCurrTime();
		cout << "TSDF cost time " << (time_end - time_start)/1000.0f << endl;

		ref = curr;


		break;
	}
	case LOST:
	{
		cout << "vo has lost" << endl;
		break;
	}

	}
}

void VisualOdometry::outputModel()
{
	tsdf_build->Output();

}

VisualOdometry::~VisualOdometry()
{

}

} /* namespace myfusion */
