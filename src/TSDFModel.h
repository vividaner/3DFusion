/*
 * TSDFModel.h
 *
 *  Created on: Jul 23, 2017
 *      Author: dan
 */

#ifndef TSDFMODEL_H_
#define TSDFMODEL_H_

#include "common_include.h"
#include "Cuda/tsdf.h"
#include "Camera.h"
#include "Frame.h"

namespace myfusion {

class TSDFModel {
public:
	typedef std::shared_ptr<TSDFModel> Ptr;

	TSDFModel(int width, int height, Camera::Ptr& camera);

	void InitTSDFModel(Camera::Ptr& camera);
	void UpdateData(Frame::Ptr frame);
	void Output();
	void SaveGrid2PointCloud();

	virtual ~TSDFModel();

private:
	unsigned int voxel_grid_dim_x;
	unsigned int voxel_grid_dim_y;
	unsigned int voxel_grid_dim_z;
	float voxel_grid_origin_x;
	float voxel_grid_origin_y;
	float voxel_grid_origin_z;

	float voxel_size;
	float trunc_margin;

	unsigned int width_frame;
	unsigned int height_frame;

	float* cpu_depth_data;
	unsigned char* cpu_rgb_data;
	float* cpu_cam_K;
	float* cpu_cam2world;
	float* cpu_voxel_grid_TSDF;
	float* cpu_voxel_grid_weight;
	unsigned char* cpu_color_r;
	unsigned char* cpu_color_g;
	unsigned char* cpu_color_b;

	float* gpu_cam_K;
	float* gpu_cam2world;
	float* gpu_depth_data;
	unsigned char* gpu_rgb_data;
	float* gpu_voxel_grid_TSDF;
	float* gpu_voxel_grid_weight;
	unsigned char* gpu_color_r;
	unsigned char* gpu_color_g;
	unsigned char* gpu_color_b;

};

} /* namespace myfusion */

#endif /* TSDFMODEL_H_ */
