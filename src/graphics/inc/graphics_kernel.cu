#ifndef GRAPHICS_KERNEL_CU
#define GRAPHICS_KERNEL_CU

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include "constant.h"

__global__
void Set2dCurveDataKernel(double* r_x, double* r_y, float2* r_plot_data, 
  uint r_num)
{
  volatile uint indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    volatile double x = r_x[indx];
    volatile double y = r_y[indx];
    r_plot_data[indx] = make_float2((float)x, (float)y);
  }
}

__global__
void Set2dCurveSemiLogDataKernel(double* r_x, uint* r_y, float2* r_plot_data, 
  uint r_num)
{
  volatile uint indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    volatile double x = r_x[indx];
    volatile uint y = r_y[indx];
    volatile float y_out = (y == 0) ? -1.0 : log2f((float)y);
    r_plot_data[indx] = make_float2((float)x, y_out);
  }
}

__global__
void FindFirstSurvivedParticle(double* r_x, double* r_y, uint* r_loss, 
                               float2* r_loss_val, uint r_num, uint* r_found)
{
  volatile uint i = 0;
  while (i < r_num && r_loss[i] != 0)    
    ++i;
  if(i < r_num)
  {
    volatile double x = r_x[i];
    volatile double y = r_y[i];
    r_loss_val[0] = make_float2((float)x, (float)y);
    r_found[0] = 1;
  }
}

__global__
void Set2dPhaseSpaceDataKernel(double* r_x, double* r_y, uint* r_loss, 
  float2* r_plot_data, double* r_tmp_x, double* r_tmp_y, uint r_num)
{
  uint indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    if(r_loss[indx] == 0) // not lost
    {
      double x = r_x[indx];
      double y = r_y[indx];
      r_plot_data[indx] = make_float2((float)x, (float)y);
      r_tmp_x[indx] = x;
      r_tmp_y[indx] = y;
    }
    else // lost
    {
      r_plot_data[indx] = make_float2(0.0, 0.0); 
      r_tmp_x[indx] = 0.0;
      r_tmp_y[indx] = 0.0;
    }
  }
}

__global__
void Print(double* r_x, double* r_y, int* r_loss, uint r_num)
{
  for(uint i = 0; i < 20; ++i)
    printf("indx=%d, (%f, %f)\t%d\n", i, r_x[i], r_y[i], r_loss[i]);
}

__global__
void BlockMaxMin2DKernel(double* r_x, double* r_y, double* r_partial_x, 
                   double* r_partial_y, uint r_num, uint flag, uint* r_loss)
{
  extern __shared__ double shmem[];
  volatile uint indx = blockIdx.x*blockDim.x + threadIdx.x;
  if(!flag) // first time
  {
    // Fill the extra seats with r_x[0], this won't affect the final
    // max & min results
    if (r_loss == NULL)
    {
      shmem[threadIdx.x] = (indx<r_num) ? r_x[indx] : r_x[0]; //xmax
      shmem[threadIdx.x + blockDim.x] = 
	(indx<r_num) ? r_x[indx] : r_x[0]; //xmin
      shmem[threadIdx.x + 2 * blockDim.x] = 
	(indx<r_num) ? r_y[indx] : r_y[0]; //ymax
      shmem[threadIdx.x + 3 * blockDim.x] = 
	(indx<r_num) ? r_y[indx] : r_y[0]; //ymin
    }
    else
    {
      shmem[threadIdx.x] = (indx<r_num && r_loss[indx] == 0 ) ? r_x[indx] : 
	r_x[0]; //xmax
      shmem[threadIdx.x + blockDim.x] = (indx < r_num && r_loss[indx] == 0) ? 
	r_x[indx] : r_x[0]; //xmin  
      shmem[threadIdx.x + 2 * blockDim.x] = 
	(indx < r_num && r_loss[indx] == 0) ? r_y[indx] : r_y[0]; //ymax
      shmem[threadIdx.x + 3 * blockDim.x] = 
	(indx < r_num && r_loss[indx] == 0) ? r_y[indx] : r_y[0]; //ymin
    }
  }
  else
  {
    shmem[threadIdx.x] = (indx<r_num) ? r_x[indx] : r_x[0]; //xmax
    shmem[threadIdx.x + blockDim.x] = (indx < r_num) ? 
      r_x[indx + 2 + blockDim.x] : r_x[0]; //xmin
    shmem[threadIdx.x + 2 * blockDim.x] = (indx < r_num) ? r_y[indx] : 
      r_y[0]; //ymax  
    shmem[threadIdx.x + 3 * blockDim.x] = (indx < r_num) ? 
      r_y[indx + 2 + blockDim.x] : r_y[0]; //ymin
  }
  __syncthreads();
  
  for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      if(!isnan(shmem[threadIdx.x + offset])) // xmax
      {
        if(isnan(shmem[threadIdx.x]) || shmem[threadIdx.x] < 
           shmem[threadIdx.x + offset])
          shmem[threadIdx.x] = shmem[threadIdx.x + offset];
      }
      if(!isnan(shmem[threadIdx.x + blockDim.x + offset])) // xmin
      {
        if(isnan(shmem[threadIdx.x + blockDim.x]) || 
	  shmem[threadIdx.x + blockDim.x] > 
	  shmem[threadIdx.x + blockDim.x + offset])
          shmem[threadIdx.x + blockDim.x] = 
	    shmem[threadIdx.x + blockDim.x + offset];
      }
      if(!isnan(shmem[threadIdx.x + 2 * blockDim.x + offset])) // ymax
      {
        if(isnan(shmem[threadIdx.x + 2 * blockDim.x]) || 
	  shmem[threadIdx.x + 2 * blockDim.x] < 
	  shmem[threadIdx.x + 2 * blockDim.x + offset])
          shmem[threadIdx.x + 2 * blockDim.x] = 
	    shmem[threadIdx.x + 2 * blockDim.x + offset];
      }
      if(!isnan(shmem[threadIdx.x + 3 * blockDim.x+offset])) // ymin
      {
        if(isnan(shmem[threadIdx.x + 3 * blockDim.x]) || 
	  shmem[threadIdx.x + 3 * blockDim.x] >
          shmem[threadIdx.x + 3 * blockDim.x + offset])
          shmem[threadIdx.x + 3 * blockDim.x] = 
	    shmem[threadIdx.x + 3 * blockDim.x + offset];
      }
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!flag) // first time
    {
      r_partial_x[blockIdx.x] = shmem[0]; // xmax
      r_partial_x[blockIdx.x + 2 + gridDim.x] = shmem[blockDim.x]; // xmin
      r_partial_y[blockIdx.x] = shmem[2 * blockDim.x]; // ymax
      r_partial_y[blockIdx.x + 2 + gridDim.x] = shmem[3 * blockDim.x]; // ymin
    }
    else
    {
      r_partial_x[blockIdx.x] = shmem[0]; //xmax
      r_partial_x[blockIdx.x + 1] = shmem[blockDim.x]; //xmin
      r_partial_y[blockIdx.x] = shmem[2 * blockDim.x]; //ymax
      r_partial_y[blockIdx.x + 1] = shmem[3 * blockDim.x]; //ymin
    }
  } 
}

__global__
void BlockMaxMin1DKernel(double* r_x, uint* r_loss, double* r_partial_x, 
  uint r_num, uint flag)
{
  extern __shared__ double shmem[];
  volatile uint indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(!flag) // first time
  {
    // Fill the extra seats with r_x[0], this won't affect the final
    // max & min results
    shmem[threadIdx.x] = (indx < r_num && r_loss[indx] == 0) ? r_x[indx] : 
      r_x[0]; //xmax
    shmem[threadIdx.x + blockDim.x] = (indx < r_num && r_loss[indx] == 0) ? 
      r_x[indx] : r_x[0]; //xmin  
  }
  else
  {
    shmem[threadIdx.x] = (indx < r_num) ? r_x[indx] : r_x[0]; //xmax
    shmem[threadIdx.x + blockDim.x] = (indx < r_num) ? 
      r_x[indx + 2 + blockDim.x] : r_x[0]; //xmin
  }
  __syncthreads();
  
  for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      if(!isnan(shmem[threadIdx.x + offset])) // xmax
      {
        if(isnan(shmem[threadIdx.x]) || shmem[threadIdx.x] < 
           shmem[threadIdx.x + offset])
          shmem[threadIdx.x] = shmem[threadIdx.x + offset];
      }
      if(!isnan(shmem[threadIdx.x + blockDim.x + offset])) // xmin
      {
        if(isnan(shmem[threadIdx.x + blockDim.x]) || 
	  shmem[threadIdx.x + blockDim.x] > 
          shmem[threadIdx.x + blockDim.x + offset])
          shmem[threadIdx.x + blockDim.x] = 
	    shmem[threadIdx.x + blockDim.x + offset];
      }
     }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!flag) // first time
    {
      r_partial_x[blockIdx.x] = shmem[0]; // xmax
      r_partial_x[blockIdx.x + 2 + gridDim.x] = shmem[blockDim.x]; // xmin
    }
    else
    {
      r_partial_x[blockIdx.x] = shmem[0]; //xmax
      r_partial_x[blockIdx.x + 1] = shmem[blockDim.x]; //xmin
    }
  } 
}

__global__
void HistogramKernel(double* r_x, uint* r_input_loss, uint r_part_num, 
              uint* r_partial, uint r_bin_num, double r_min, double r_max)
{
  extern __shared__ uint shmem1[];
  volatile uint indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadIdx.x < r_bin_num)
    shmem1[threadIdx.x] = 0;
  __syncthreads();
  if(indx < r_part_num && r_input_loss[indx] == 0)
  {
    volatile double x = r_x[indx];
    volatile double bin_width = (r_max - r_min) / r_bin_num;
    volatile int bin_indx = (uint)floorf((x - r_min) / bin_width);
    if(bin_indx >= 0 && bin_indx  < r_bin_num - 1)
      atomicAdd(shmem1 + bin_indx, 1);
    else if(bin_indx < 0)
      atomicAdd(shmem1, 1);
    else if(bin_indx > r_bin_num - 1)
      atomicAdd(shmem1 + r_bin_num - 1, 1);
  }
  __syncthreads();
  if(threadIdx.x < r_bin_num)
    r_partial[blockIdx.x * r_bin_num + threadIdx.x] = shmem1[threadIdx.x];
}

__global__
void HistogramReduceKernel(uint* r_partial, uint* r_hist, uint r_bin_num)
{
  extern __shared__ uint shmem2[];
  shmem2[threadIdx.x] = r_partial[threadIdx.x * r_bin_num + blockIdx.x];
  __syncthreads();
  for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
      shmem2[threadIdx.x] += shmem2[threadIdx.x + offset];
    __syncthreads();
  }
  if(threadIdx.x == 0)
    r_hist[blockIdx.x] = shmem2[0];
}

__global__
void Set3dDataKernel(double* r_x, double* r_y, double* r_z, 
  float4* r_plot_data, uint r_num)
{
  volatile uint indx = blockIdx.x * blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    double x = r_x[indx];
    double y = r_y[indx];
    double z = r_z[indx] * 0.005;
    r_plot_data[indx] = make_float4((float)x, (float)y, (float)z, 0.01);
  }
}

__global__
void Histogram2DKernel(uint* r_hist, double* r_x, double* r_y, uint* r_loss, 
  uint r_part_num, double r_xmin, double r_xmax, double r_ymin, double r_ymax, 
  uint r_bin_num_x, uint r_bin_num_y)
{
  uint np = threadIdx.x + blockIdx.x * blockDim.x;  
  uint stride = blockDim.x * gridDim.x;
  while(np < r_part_num)
  {
    double x = r_x[np];
    double y = r_y[np];
    if (r_loss[np] == 0 && x >= r_xmin && x <= r_xmax && y >= r_ymin && 
      y <= r_ymax)
    {
      double dx = (r_xmax - r_xmin) / r_bin_num_x;
      double dy = (r_ymax - r_ymin) / r_bin_num_y;
      uint indx_x = (uint) floorf((x - r_xmin) / dx);
      uint indx_y = (uint) floorf((y - r_ymin) / dy);
      if (indx_x > r_bin_num_x - 1) 
        indx_x = r_bin_num_x - 1;
      if (indx_y > r_bin_num_y - 1) 
        indx_y = r_bin_num_y - 1;
      uint indx = indx_y * r_bin_num_x + indx_x;
      atomicAdd(r_hist + indx, 1);
    }
    np += stride;
  } 
} 

__global__
void SetHistogram2DCoordinateDataKernel(float2* r_data, double r_xmin, 
  double r_xmax, double r_ymin, double r_ymax, uint r_bin_num_x, 
  uint r_bin_num_y)
{
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < r_bin_num_x * r_bin_num_y)
  {
    uint idy = tid / r_bin_num_x;
    uint idx = tid % r_bin_num_x;
    uint indx = idy * r_bin_num_x + idx;
    double dx = (r_xmax - r_xmin) / r_bin_num_x;
    double dy = (r_ymax - r_ymin) / r_bin_num_y;
    double x = r_xmin + dx * (0.5 + idx);
    double y = r_ymin + dy * (0.5 + idy);
    r_data[indx] = make_float2((float)x, (float)y);
  }
}

#endif
