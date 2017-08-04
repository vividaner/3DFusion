#include "tsdf.h"

void FatalError(const int lineNumber = 0) {
  std::cerr << "FatalError";
  if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
  std::cerr << ". Program Terminated." << std::endl;
  cudaDeviceReset();
  exit(EXIT_FAILURE);
}

void checkCUDA(const int lineNumber, cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
    FatalError();
  }
}

__global__
void Integrate(float* cam_K, float* cam2base, float* depth_im, unsigned char* rgb_im,
				unsigned char* voxel_grid_color_r, unsigned char* voxel_grid_color_g, unsigned char* voxel_grid_color_b,
				float* voxel_grid_TSDF, float* voxel_grid_weight,
				unsigned int im_width, unsigned int im_height, 
                unsigned int voxel_grid_dim_x, unsigned int voxel_grid_dim_y, unsigned int voxel_grid_dim_z,
				float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
			    float voxel_size, float trunc_margin)
{
    int pt_grid_z = blockIdx.x;
    int pt_grid_y = threadIdx.x;

    for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
    float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
    int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];
    

    if (depth_val <= 0 || depth_val > 6)
      continue;

    float diff = (depth_val - pt_cam_z) * sqrtf(1 + powf((pt_cam_x / pt_cam_z), 2) + powf((pt_cam_y / pt_cam_z), 2));
    if (diff <= -trunc_margin)
      continue;
      
    char color_r = rgb_im[pt_pix_y * im_width * 3 + pt_pix_x];
	char color_g = rgb_im[pt_pix_y * im_width * 3 + pt_pix_x + 1];
	char color_b = rgb_im[pt_pix_y * im_width * 3 + pt_pix_x + 2];

    // Integrate
    int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
    
    int ori_color_r = voxel_grid_color_r[volume_idx];
	int ori_color_g = voxel_grid_color_r[volume_idx];
	int ori_color_b = voxel_grid_color_r[volume_idx];
				
	int integrate_color_r = color_r;
	int integrate_color_g = color_g;
	int integrate_color_b = color_b;
				
	int update_color_r = (ori_color_r * weight_old + integrate_color_r)/weight_new;
	int update_color_g = (ori_color_g * weight_old + integrate_color_g)/weight_new;
	int update_color_b = (ori_color_b * weight_old + integrate_color_b)/weight_new;
				
	voxel_grid_color_r[volume_idx] = min(max(update_color_r, 0), 255);
	voxel_grid_color_g[volume_idx] = min(max(update_color_g, 0), 255);
	voxel_grid_color_b[volume_idx] = min(max(update_color_b, 0), 255);
    
  }
}

void TSDFModelIntegrate(float* gpu_cam_K, float* gpu_cam2world, float* gpu_depth_data, unsigned char* gpu_rgb_data,
						float* gpu_voxel_grid_TSDF, float* gpu_voxel_grid_weight,
						unsigned char* gpu_color_r, unsigned char* gpu_color_g, unsigned char* gpu_color_b,
						unsigned int width_frame, unsigned int height_frame,
						unsigned int voxel_grid_dim_x, unsigned int voxel_grid_dim_y, unsigned int voxel_grid_dim_z,
						float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z,
				        float voxel_size, float trunc_margin)
{


	checkCUDA(__LINE__, cudaGetLastError());
	Integrate<<<voxel_grid_dim_z, voxel_grid_dim_y>>>(gpu_cam_K, gpu_cam2world, gpu_depth_data, gpu_rgb_data,
													  gpu_color_r, gpu_color_g, gpu_color_b,
													  gpu_voxel_grid_TSDF, gpu_voxel_grid_weight,
													  width_frame, height_frame, 
													  voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
													  voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
													  voxel_size, trunc_margin);
	cudaSafeCall(cudaThreadSynchronize());
}

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
				   unsigned int voxel_grid_dim_z)
{
	checkCUDA(__LINE__, cudaGetLastError());
	
	cudaSafeCall(cudaMalloc(&gpu_depth_data, width*height*sizeof(float)));
	cudaSafeCall(cudaMalloc(&gpu_rgb_data, width*height*sizeof(unsigned char)*3));
	cudaSafeCall(cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float)));
	cudaSafeCall(cudaMalloc(&gpu_cam2world, 4 * 4 * sizeof(float)));
	cudaSafeCall(cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float)));
	cudaSafeCall(cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float)));
	cudaSafeCall(cudaMalloc(&gpu_color_r, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char)));
	cudaSafeCall(cudaMalloc(&gpu_color_g, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char)));
	cudaSafeCall(cudaMalloc(&gpu_color_b, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char)));
	
	
	checkCUDA(__LINE__, cudaGetLastError());
}

void TSDFInitDataToGPU(float* cpu_cam_K, float* cpu_voxel_grid_TSDF, float* cpu_voxel_grid_weight,
						  float* gpu_cam_K, float* gpu_voxel_grid_TSDF, float* gpu_voxel_grid_weight,
						  unsigned int voxel_grid_dim_x, unsigned int voxel_grid_dim_y, unsigned int voxel_grid_dim_z)
{
	cudaSafeCall(cudaMemcpy(gpu_cam_K, cpu_cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
	
	cudaSafeCall(cudaMemcpy(gpu_voxel_grid_TSDF, cpu_voxel_grid_TSDF, 
	voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice));
	
	cudaSafeCall(cudaMemcpy(gpu_voxel_grid_weight, cpu_voxel_grid_weight, 
	voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice));
}

void TSDFUpdateDataToGPU(float* cpu_cam2world, float* cpu_depth_data, unsigned char* cpu_rgb_data,
						 float* gpu_cam2world, float* gpu_depth_data, unsigned char* gpu_rgb_data,
							unsigned int width, unsigned int height)
{
	cudaSafeCall(cudaMemcpy(gpu_cam2world, cpu_cam2world, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(gpu_depth_data, cpu_depth_data, width * height * sizeof(float), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(gpu_rgb_data, cpu_rgb_data, width * height * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice));
	
}

void TSDFGetDataFromGPU(float* cpu_voxel_grid_TSDF, float* cpu_voxel_grid_weight,
		                unsigned char* cpu_color_r, unsigned char* cpu_color_g, unsigned char* cpu_color_b,
		   	   	   	   	float* gpu_voxel_grid_TSDF, float* gpu_voxel_grid_weight,
		   	   	   	   	unsigned char* gpu_color_r, unsigned char* gpu_color_g, unsigned char* gpu_color_b,
		   	   	   	   	unsigned int voxel_grid_dim_x, unsigned int voxel_grid_dim_y, unsigned int voxel_grid_dim_z)
{
	cudaSafeCall(cudaMemcpy(cpu_voxel_grid_TSDF, gpu_voxel_grid_TSDF, 
			   voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost));
	
	cudaSafeCall(cudaMemcpy(cpu_voxel_grid_weight, gpu_voxel_grid_weight, 
	           voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost));
	           
	cudaSafeCall(cudaMemcpy(cpu_color_r, gpu_color_r,
	             voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	             
	cudaSafeCall(cudaMemcpy(cpu_color_g, gpu_color_g,
	             voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char), cudaMemcpyDeviceToHost));
	             
	cudaSafeCall(cudaMemcpy(cpu_color_b, gpu_color_b,
	             voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}		   	   	   	   	