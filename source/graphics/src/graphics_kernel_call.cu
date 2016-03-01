#include "graphics_kernel.cu"
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
/*
#include <thrust/reduce.h>
*/

extern "C"
{
  void Set2dCurveData(uint r_gridsz, uint r_blcksz, double* r_x, double* r_y, float2* r_plot_data, uint r_num, double2* r_last_data)
  {
    Set2dCurveDataKernel<<<r_gridsz, r_blcksz>>>(r_x, r_y, r_plot_data, r_num);
    cudaMemcpy(&(r_last_data->x), r_x+r_num-1,sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(r_last_data->y), r_y+r_num-1,sizeof(double), cudaMemcpyDeviceToHost);
  }

  void Set2dHistogramData(uint r_gridsz, uint r_blcksz, double* r_x, uint* r_y, float2* r_plot_data, uint r_num)
  {
    Set2dCurveSemiLogDataKernel<<<r_gridsz, r_blcksz>>>(r_x, r_y, r_plot_data, r_num);
  }

  void SetPhaseSpaceData(uint r_gridsz, uint r_blcksz, double* r_x, 
                         double* r_y, uint* r_loss, double* r_tmp_x,
                         double* r_tmp_y, uint* r_tmp_loss,
                         float2* r_plot_data, uint r_num)
  {
    // look for the first survived particle coordinates and use it as the default value
/*
    float2* loss_val;
    cudaMalloc((void**)&loss_val, sizeof(float2));
    uint* found; // found ->1;
    cudaMalloc((void**)&found, sizeof(uint));
    cudaMemset(found, 0, sizeof(uint));
    FindFirstSurvivedParticle<<<1, 1>>>(r_x, r_y, r_loss, loss_val, r_num, found);
    uint* h_found = new uint;
    cudaMemcpy(h_found, found, sizeof(uint), cudaMemcpyDeviceToHost);
    float2* h_loss_val = new float2;
    cudaMemcpy(h_loss_val, loss_val, sizeof(float2), cudaMemcpyDeviceToHost);
    delete h_loss_val;
    delete h_found;
*/
    // Now if the particle is lost, its x and y is replaced with 0.0, just like the reference particle.
    Set2dPhaseSpaceDataKernel<<<r_gridsz, r_blcksz>>>(r_x, r_y, r_loss, r_plot_data, r_tmp_x, r_tmp_y, r_num/*, loss_val*/);

//    cudaFree(loss_val);
//    cudaFree(found);
  }

  void FindMaxMin2D(uint r_gridsz, uint r_blcksz, double* r_x, double* r_y, 
                  double* r_partial_x, double* r_partial_y, double4* r_maxmin, 
                  /*double4* r_maxmin_d,*/ uint r_num, uint* r_loss)
  {
    if (r_loss == NULL)
    {
      BlockMaxMin2DKernel<<<r_gridsz, r_blcksz, 4*r_blcksz*sizeof(double)>>>
        (r_x, r_y, r_partial_x, r_partial_y, r_num, 0, NULL);

      BlockMaxMin2DKernel<<<1, r_gridsz, 4*r_gridsz*sizeof(double)>>>
        (r_partial_x, r_partial_y, r_partial_x+r_gridsz, r_partial_y+r_gridsz,
         r_gridsz, 1, NULL);
    }
    else
    {
      std::cout << "FindMaxMin2D: not null!" << std::endl;
      BlockMaxMin2DKernel<<<r_gridsz, r_blcksz, 4*r_blcksz*sizeof(double)>>>
        (r_x, r_y, r_partial_x, r_partial_y, r_num, 0, r_loss);

      BlockMaxMin2DKernel<<<1, r_gridsz, 4*r_gridsz*sizeof(double)>>>
        (r_partial_x, r_partial_y, r_partial_x+r_gridsz, r_partial_y+r_gridsz,
         r_gridsz, 1, NULL);
    }

    cudaMemcpy(&(r_maxmin->x), r_partial_x+r_gridsz, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(r_maxmin->y), r_partial_x+r_gridsz+1, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(r_maxmin->z), r_partial_y+r_gridsz, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(r_maxmin->w), r_partial_y+r_gridsz+1, sizeof(double), cudaMemcpyDeviceToHost);
  }

  void FindMaxMin1D(uint r_gridsz, uint r_blcksz, double* r_x, uint* r_loss,
                    double* r_partial_x, double4* r_maxmin, uint r_num)
  {
    BlockMaxMin1DKernel<<<r_gridsz, r_blcksz, 2*r_blcksz*sizeof(double)>>>
      (r_x, r_loss, r_partial_x, r_num, 0); 
    BlockMaxMin1DKernel<<<1, r_gridsz, 2*r_gridsz*sizeof(double)>>>
      (r_partial_x, NULL, r_partial_x+r_gridsz, r_gridsz, 1); 
    cudaMemcpy(&(r_maxmin->x), r_partial_x+r_gridsz, sizeof(double), cudaMemcpyDeviceToHost); // xmax
    cudaMemcpy(&(r_maxmin->y), r_partial_x+r_gridsz+1, sizeof(double), cudaMemcpyDeviceToHost); // xmin
  }

  void UpdateHistogram(double* r_input_y, uint* r_input_loss, uint* r_hist, 
                       uint* r_partial_hist, uint r_data_num, uint r_bin_num, 
                       double r_min, double r_max, uint r_gridsz, uint r_blksz)
  {
    HistogramKernel<<<r_gridsz, r_blksz, sizeof(uint)*r_bin_num>>>(r_input_y, 
            r_input_loss, r_data_num, r_partial_hist, r_bin_num, r_min, r_max);
    HistogramReduceKernel<<<r_bin_num, r_gridsz, sizeof(uint)*r_gridsz>>>(
                    r_partial_hist, r_hist, r_bin_num);
  }
  void Set3dData(uint r_gridsz, uint r_blcksz, double* r_x, double* r_y, 
                double* r_z, float4* r_plot_data, uint r_num)
  {
    Set3dDataKernel<<<r_gridsz, r_blcksz>>>(r_x, r_y, r_z, r_plot_data, r_num);
  }

  void UpdateHistogram2DKernelCall(uint* r_hist, double* r_input_x, double* r_input_y, 
    uint* r_input_loss, uint r_data_num, double4* r_maxmin, uint r_bin_num_x, uint r_bin_num_y,
    uint r_thrd_num, uint r_blck_num)
  {
    Histogram2DKernel<<<r_blck_num, r_thrd_num>>>(r_hist, r_input_x, r_input_y, r_input_loss, 
      r_data_num, r_maxmin->y, r_maxmin->x, r_maxmin->w, r_maxmin->z, r_bin_num_x, r_bin_num_y);  
  }

  void SetHistogram2DCoordinateDataKernelCall(float2* r_data, double4* r_maxmin, uint r_bin_num_x, 
    uint r_bin_num_y, uint r_thrd_num, uint r_blck_num)
  {
    SetHistogram2DCoordinateDataKernel<<<r_blck_num, r_thrd_num>>>(r_data, r_maxmin->y, r_maxmin->x,
      r_maxmin->w, r_maxmin->z, r_bin_num_x, r_bin_num_y);
  }
} // extern "C"
