/*
 * TSDFModel.cpp
 *
 *  Created on: Jul 23, 2017
 *      Author: dan
 */

#include "TSDFModel.h"


namespace myfusion {

template <class T>
void safefree(T* data)
{
	if(data)
	{
		delete[] data;
		data = nullptr;
	}
}

template <class T>
void safecudafree(T* data)
{
	if(data)
	{
		cudaFree(data);
		data = nullptr;
	}
}

TSDFModel::TSDFModel(int width, int height, Camera::Ptr& camera) :
		width_frame(width),
		height_frame(height),
		voxel_grid_dim_x(VOXEL_GRID_SIZE),
		voxel_grid_dim_y(VOXEL_GRID_SIZE),
		voxel_grid_dim_z(VOXEL_GRID_SIZE)
{
	voxel_grid_origin_x = -1.5f;
	voxel_grid_origin_y = -1.5f;
	voxel_grid_origin_z = 0.5f;

	voxel_size = 0.006f;
	trunc_margin = voxel_size*5;

	cpu_depth_data = nullptr;
	cpu_rgb_data = nullptr;
	cpu_cam_K = nullptr;
	cpu_cam2world = nullptr;
	cpu_voxel_grid_TSDF = nullptr;
	cpu_voxel_grid_weight = nullptr;
	cpu_color_r = nullptr;
	cpu_color_g = nullptr;
	cpu_color_b = nullptr;

	gpu_cam_K = nullptr;
	gpu_cam2world = nullptr;
	gpu_depth_data = nullptr;
	gpu_rgb_data = nullptr;
	gpu_voxel_grid_TSDF = nullptr;
	gpu_voxel_grid_weight = nullptr;
	gpu_color_r = nullptr;
	gpu_color_g = nullptr;
	gpu_color_b = nullptr;
}

void TSDFModel::InitTSDFModel(Camera::Ptr& camera)
{
	cpu_depth_data = new float[width_frame * height_frame];
	cpu_rgb_data = new unsigned char[width_frame * height_frame * 3];
	cpu_voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
	cpu_voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
	cpu_cam_K = new float[3 * 3];
	cpu_cam2world = new float[4 * 4];
	cpu_color_r = new unsigned char[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
	cpu_color_g = new unsigned char[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
	cpu_color_b = new unsigned char[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];

	for(unsigned int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
		cpu_voxel_grid_TSDF[i] = 1.0f;
	memset(cpu_voxel_grid_weight, 0, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
	memset(cpu_color_r, 0, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char));
	memset(cpu_color_g, 0, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char));
	memset(cpu_color_b, 0, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char));

	cpu_cam_K[0] = camera->fx_;
	cpu_cam_K[1] = 0.0f;
	cpu_cam_K[2] = camera->cx_;
	cpu_cam_K[3] = 0;
	cpu_cam_K[4] = camera->fy_;
	cpu_cam_K[5] = camera->cy_;
	cpu_cam_K[6] = 0;
	cpu_cam_K[7] = 0;
	cpu_cam_K[8] = 1.0f;


	TSDFMemoryInit(gpu_depth_data,
				   gpu_rgb_data,
				   gpu_voxel_grid_TSDF,
				   gpu_voxel_grid_weight,
				   gpu_color_r,
				   gpu_color_g,
				   gpu_color_b,
				   gpu_cam_K,
				   gpu_cam2world,
				   width_frame,
				   height_frame,
				   voxel_grid_dim_x,
				   voxel_grid_dim_y,
				   voxel_grid_dim_z);

	TSDFInitDataToGPU(cpu_cam_K, cpu_voxel_grid_TSDF, cpu_voxel_grid_weight,
						  gpu_cam_K, gpu_voxel_grid_TSDF, gpu_voxel_grid_weight,
						  voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);




}

void TSDFModel::UpdateData(Frame::Ptr frame)
{
	for(unsigned int i = 0; i < 4; ++i)
	{
		for(unsigned int j = 0; j < 4; ++j)
		{
			cpu_cam2world[i * 4 + j] = frame->T_w_c.cast<float>().matrix()(i,j);
		}
	}

	unsigned short* depthrun = (unsigned short*)frame->getDepthData();
	for(unsigned int i = 0; i < width_frame * height_frame; i++)
	{
		cpu_depth_data[i] = depthrun[i] / 1000.0f;
	}

	memcpy(cpu_rgb_data, frame->getTextureData(), width_frame * height_frame * 3);

	TSDFUpdateDataToGPU(cpu_cam2world, cpu_depth_data, cpu_rgb_data,
						gpu_cam2world, gpu_depth_data, gpu_rgb_data,
								width_frame, height_frame);

	cout << "test here" << endl;

	TSDFModelIntegrate(gpu_cam_K, gpu_cam2world, gpu_depth_data, gpu_rgb_data,
			           gpu_voxel_grid_TSDF, gpu_voxel_grid_weight,
			           gpu_color_r, gpu_color_g, gpu_color_b,
						width_frame, height_frame, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
						voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
						voxel_size, trunc_margin);

	cout << "test here" << endl;

}

void TSDFModel::SaveGrid2PointCloud()
{
	// Count total number of points in point cloud
	int num_pts = 0;
	float tsdf_thresh = 0.2f;
	float weight_thresh = 1.0f;
	string file_name = "tsdf.ply";

	for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z;
			i++)
		if (std::abs(cpu_voxel_grid_TSDF[i]) < tsdf_thresh
				&& cpu_voxel_grid_weight[i] > weight_thresh)
			num_pts++;

	// Create header for .ply file
	FILE *fp = fopen(file_name.c_str(), "w");
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", num_pts);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "property uchar red\n");
	fprintf(fp, "property uchar green\n");
	fprintf(fp, "property uchar blue\n");
	fprintf(fp, "end_header\n");

	// Create point cloud content for ply file
	for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; i++)
	{

		// If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
		if (std::abs(cpu_voxel_grid_TSDF[i]) < tsdf_thresh
				&& cpu_voxel_grid_weight[i] > weight_thresh) {

			// Compute voxel indices in int for higher positive number range
			int z = floor(i / (voxel_grid_dim_x * voxel_grid_dim_y));
			int y = floor((i - (z * voxel_grid_dim_x * voxel_grid_dim_y)) / voxel_grid_dim_x);
			int x = i - (z * voxel_grid_dim_x * voxel_grid_dim_y) - (y * voxel_grid_dim_x);

			char r = cpu_color_r[i];
			char g = cpu_color_g[i];
			char b = cpu_color_b[i];

			// Convert voxel indices to float, and save coordinates to ply file
			float pt_base_x = voxel_grid_origin_x + (float) x * voxel_size;
			float pt_base_y = voxel_grid_origin_y + (float) y * voxel_size;
			float pt_base_z = voxel_grid_origin_z + (float) z * voxel_size;
			fwrite(&pt_base_x, sizeof(float), 1, fp);
			fwrite(&pt_base_y, sizeof(float), 1, fp);
			fwrite(&pt_base_z, sizeof(float), 1, fp);
			fwrite(&r, sizeof(char), 1, fp);
			fwrite(&g, sizeof(char), 1, fp);
			fwrite(&b, sizeof(char), 1, fp);
		}
	}
	fclose(fp);
}

void TSDFModel::Output()
{
	TSDFGetDataFromGPU(cpu_voxel_grid_TSDF, cpu_voxel_grid_weight,
					   cpu_color_r, cpu_color_g, cpu_color_b,
					   gpu_voxel_grid_TSDF, gpu_voxel_grid_weight,
					   gpu_color_r, gpu_color_g, gpu_color_b,
					   voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z);
	SaveGrid2PointCloud();
}


TSDFModel::~TSDFModel() {
	safefree(cpu_cam_K);
	safefree(cpu_cam2world);
	safefree(cpu_voxel_grid_TSDF);
	safefree(cpu_voxel_grid_weight);
	safefree(cpu_depth_data);
	safefree(cpu_color_r);
	safefree(cpu_color_g);
	safefree(cpu_color_b);

	safecudafree(gpu_cam_K);
	safecudafree(gpu_cam2world);
	safecudafree(gpu_voxel_grid_TSDF);
	safecudafree(gpu_voxel_grid_weight);
	safecudafree(gpu_depth_data);
}

} /* namespace myfusion */
