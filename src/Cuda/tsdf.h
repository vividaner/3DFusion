/*
 * tsdf.h
 *
 *  Created on: Jul 24, 2017
 *      Author: dan
 */

#ifndef TSDF_H_
#define TSDF_H_

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "containers/safe_call.hpp"

#include <vector>
#include <iostream>



void TSDFMemoryInit(float* &gpu_depth_data,
				   unsigned char* &gpu_rgb_data,
				   float* &gpu_voxel_grid_TSDF,
				   float* &gpu_voxel_grid_weight,
				   unsigned char* &gpu_color_r,
				   unsigned char* &gpu_color_g,
				   unsigned char* &gpu_color_b,
				   float* &gpu_cam_K,
				   float* &gpu_cam2world,
				   unsigned int width,
				   unsigned int height,
				   unsigned int voxel_grid_dim_x,
				   unsigned int voxel_grid_dim_y,
				   unsigned int voxel_grid_dim_z);

void TSDFInitDataToGPU(float* cpu_cam_K, float* cpu_voxel_grid_TSDF, float* cpu_voxel_grid_weight,
						  float* gpu_cam_K, float* gpu_voxel_grid_TSDF, float* gpu_voxel_grid_weight,
						  unsigned int voxel_grid_dim_x, unsigned int voxel_grid_dim_y, unsigned int voxel_grid_dim_z);


void TSDFUpdateDataToGPU(float* cpu_cam2world, float* cpu_depth_data, unsigned char* cpu_rgb_data,
						 float* gpu_cam2world, float* gpu_depth_data, unsigned char* gpu_rgb_data,
							unsigned int width, unsigned int height);

void TSDFModelIntegrate(float* gpu_cam_K, float* gpu_cam2world,
						float* gpu_depth_data, unsigned char* gpu_rgb_data,
						float* gpu_voxel_grid_TSDF, float* gpu_voxel_grid_weight,
						unsigned char* gpu_color_r, unsigned char* gpu_color_g, unsigned char* gpu_color_b,
						unsigned int width_frame, unsigned int height_frame,
						unsigned int voxel_grid_dim_x, unsigned int voxel_grid_dim_y, unsigned int voxel_grid_dim_z,
						float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
				        float voxel_size, float trunc_margin);

void TSDFGetDataFromGPU(float* cpu_voxel_grid_TSDF, float* cpu_voxel_grid_weight,
		                unsigned char* cpu_color_r, unsigned char* cpu_color_g, unsigned char* cpu_color_b,
		   	   	   	   	float* gpu_voxel_grid_TSDF, float* gpu_voxel_grid_weight,
		   	   	   	   	unsigned char* gpu_color_r, unsigned char* gpu_color_g, unsigned char* gpu_color_b,
		   	   	   	   	unsigned int voxel_grid_dim_x, unsigned int voxel_grid_dim_y, unsigned int voxel_grid_dim_z);


#endif /* TSDF_H_ */
