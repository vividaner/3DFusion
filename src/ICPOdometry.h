/*
 * ICPOdometry.h
 *
 *  Created on: Jul 20, 2017
 *      Author: dan
 */

#ifndef ICPODOMETRY_H_
#define ICPODOMETRY_H_

#include "Cuda/internal.h"
#include "common_include.h"
#include "Camera.h"

#include <vector>
#include <sophus/se3.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

namespace myfusion {

class ICPOdometry {
public:
	    typedef std::shared_ptr<ICPOdometry> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        ICPOdometry(int width,
                     int height,
                     Camera::Ptr& camera,
                     float distThresh = 0.10f,
                     float angleThresh = sinf(20.f * 3.14159254f / 180.f));

        virtual ~ICPOdometry();

        void initICP(unsigned short * depth, const float depthCutoff = 20.0f);

        void initICPModel(unsigned short * depth, const float depthCutoff = 20.0f);

        void getIncrementalTransformation(Sophus::SE3d & T_prev_curr, int threads, int blocks);

        float lastError;
        float lastInliers;

    private:
        std::vector<DeviceArray2D<unsigned short>> depth_tmp;

        std::vector<DeviceArray2D<float>> vmaps_prev;
        std::vector<DeviceArray2D<float>> nmaps_prev;

        std::vector<DeviceArray2D<float>> vmaps_curr;
        std::vector<DeviceArray2D<float>> nmaps_curr;

        Intr intr;

        DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> sumData;
        DeviceArray<Eigen::Matrix<float,29,1,Eigen::DontAlign>> outData;

        static const int NUM_PYRS = 3;

        std::vector<int> iterations;

        float dist_thresh;
        float angle_thresh;

        const int width;
        const int height;
        const float cx, cy, fx, fy;
};

} /* namespace myfusion */

#endif /* ICPODOMETRY_H_ */
