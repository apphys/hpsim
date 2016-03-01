#ifndef GRAPHICS_KERNEL_CU
#define GRAPHICS_KERNEL_CU

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>

__global__
void Set2dCurveDataKernel(double* r_x, double* r_y, float2* r_plot_data, uint r_num)
{
//  printf("%s\n", "Set2dDataKernel");
  volatile uint indx = blockIdx.x*blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    volatile double x = r_x[indx];
    volatile double y = r_y[indx];
    r_plot_data[indx] = make_float2((float)x, (float)y);
//      printf("blkid=%d, thrdid=%d, plot_data.x=%f, plot_data.y=%f\n", blockIdx.x, threadIdx.x, r_plot_data[indx].x, r_plot_data[indx].y);
  }
}

__global__
void Set2dCurveSemiLogDataKernel(double* r_x, uint* r_y, float2* r_plot_data, uint r_num)
{
//  printf("%s\n", "second version Kernel is called");
  volatile uint indx = blockIdx.x*blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    volatile double x = r_x[indx];
    volatile uint y = r_y[indx];
    volatile float y_out = (y==0) ? -1.0 : log2f((float)y);
    r_plot_data[indx] = make_float2((float)x, y_out);
//    printf("blkid=%d, thrdid=%d, plot_data.x=%f, plot_data.y=%f\n", blockIdx.x, threadIdx.x, r_plot_data[indx].x, r_plot_data[indx].y);
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
      float2* r_plot_data, double* r_tmp_x, double* r_tmp_y, uint r_num/*, float2* r_loss_val*/)
{
  uint indx = blockIdx.x*blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    if(r_loss[indx] == 0)
    {
      double x = r_x[indx];
      double y = r_y[indx];
      r_plot_data[indx] = make_float2((float)x, (float)y);
      r_tmp_x[indx] = x;
      r_tmp_y[indx] = y;
    }
    else
    {
      r_plot_data[indx] = make_float2(0.0, 0.0); //r_loss_val[0];
      r_tmp_x[indx] = 0.0;
      r_tmp_y[indx] = 0.0;
    }
  }
}

__global__
void Print(double* r_x, double* r_y, int* r_loss, uint r_num)
{
//  volatile uint indx = blockIdx.x*blockDim.x + threadIdx.x;
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
      shmem[threadIdx.x] = (indx<r_num) ? r_x[indx] : r_x[0];                 //xmax
      shmem[threadIdx.x+blockDim.x] = (indx<r_num) ? r_x[indx] : r_x[0];      //xmin  
      shmem[threadIdx.x+2*blockDim.x] = (indx<r_num) ? r_y[indx] : r_y[0];    //ymax
      shmem[threadIdx.x+3*blockDim.x] = (indx<r_num) ? r_y[indx] : r_y[0];    //ymin
    }
    else
    {
      shmem[threadIdx.x] = (indx<r_num && r_loss[indx] == 0 ) ? r_x[indx] : r_x[0];                 //xmax
      shmem[threadIdx.x+blockDim.x] = (indx<r_num && r_loss[indx] == 0) ? r_x[indx] : r_x[0];      //xmin  
      shmem[threadIdx.x+2*blockDim.x] = (indx<r_num && r_loss[indx] == 0) ? r_y[indx] : r_y[0];    //ymax
      shmem[threadIdx.x+3*blockDim.x] = (indx<r_num && r_loss[indx] == 0) ? r_y[indx] : r_y[0];    //ymin
    }
  }
  else
  {
    shmem[threadIdx.x] = (indx<r_num) ? r_x[indx] : r_x[0];                 //xmax
    shmem[threadIdx.x+blockDim.x] = (indx<r_num) ? r_x[indx+2+blockDim.x] : r_x[0]; //xmin
    shmem[threadIdx.x+2*blockDim.x] = (indx<r_num) ? r_y[indx] : r_y[0];    //ymax  
    shmem[threadIdx.x+3*blockDim.x] = (indx<r_num) ? r_y[indx+2+blockDim.x] : r_y[0];    //ymin
  }
  __syncthreads();
  
  for(int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      if(!isnan(shmem[threadIdx.x + offset])) // xmax
      {
        if(isnan(shmem[threadIdx.x]) || shmem[threadIdx.x] < 
           shmem[threadIdx.x + offset])
          shmem[threadIdx.x] = shmem[threadIdx.x + offset];
      }
      if(!isnan(shmem[threadIdx.x+blockDim.x+offset])) // xmin
      {
        if(isnan(shmem[threadIdx.x+blockDim.x]) || shmem[threadIdx.x+blockDim.x] > 
           shmem[threadIdx.x+blockDim.x+offset])
          shmem[threadIdx.x+blockDim.x] = shmem[threadIdx.x+blockDim.x+offset];
      }
      if(!isnan(shmem[threadIdx.x+2*blockDim.x+offset])) // ymax
      {
        if(isnan(shmem[threadIdx.x+2*blockDim.x]) || shmem[threadIdx.x+2*blockDim.x] < 
           shmem[threadIdx.x+2*blockDim.x+offset])
          shmem[threadIdx.x+2*blockDim.x] = shmem[threadIdx.x+2*blockDim.x+offset];
      }
      if(!isnan(shmem[threadIdx.x+3*blockDim.x+offset])) // ymin
      {
        if(isnan(shmem[threadIdx.x+3*blockDim.x]) || shmem[threadIdx.x+3*blockDim.x] >
           shmem[threadIdx.x+3*blockDim.x+offset])
          shmem[threadIdx.x+3*blockDim.x] = shmem[threadIdx.x+3*blockDim.x+offset];
      }
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!flag) // first time
    {
      r_partial_x[blockIdx.x] = shmem[0];                                // xmax
      r_partial_x[blockIdx.x+2+gridDim.x] = shmem[blockDim.x];          // xmin
      r_partial_y[blockIdx.x] = shmem[2*blockDim.x];                     // ymax
      r_partial_y[blockIdx.x+2+gridDim.x] = shmem[3*blockDim.x];        // ymin
    }
    else
    {
      r_partial_x[blockIdx.x] = shmem[0];                //xmax
      r_partial_x[blockIdx.x+1] = shmem[blockDim.x];     //xmin
      r_partial_y[blockIdx.x] = shmem[2*blockDim.x];     //ymax
      r_partial_y[blockIdx.x+1] = shmem[3*blockDim.x];   //ymin
    }
  } 
}

__global__
void BlockMaxMin1DKernel(double* r_x, uint* r_loss, double* r_partial_x, uint r_num, uint flag)
{
//  printf("%s\n", "BlockMaxMin1DKernel");
  extern __shared__ double shmem[];
  volatile uint indx = blockIdx.x*blockDim.x + threadIdx.x;
  if(!flag) // first time
  {
    // Fill the extra seats with r_x[0], this won't affect the final
    // max & min results
    shmem[threadIdx.x] = (indx<r_num && r_loss[indx] == 0) ? r_x[indx] : r_x[0];                 //xmax
    shmem[threadIdx.x+blockDim.x] = (indx<r_num && r_loss[indx] == 0) ? r_x[indx] : r_x[0];      //xmin  
#ifdef _DEBUG
//    printf("blkid=%d, thrdid=%d, indx=%d, r_num=%d, sm_xmax=%f, sm_xmin=%f, sm_ymax=%f, sm_ymin=%f\n", blockIdx.x, threadIdx.x, indx, r_num, shmem[threadIdx.x], shmem[threadIdx.x+blockDim.x], shmem[threadIdx.x+2*blockDim.x], shmem[threadIdx.x+3*blockDim.x]);
//    printf("blkid=%d, thrdid=%d, indx=%d, r_num=%d, xmax=%f, xmin=%f, ymax=%f, ymin=%f\n", blockIdx.x, threadIdx.x, indx, r_num, r_x[indx], r_x[indx], r_y[indx], r_y[indx]);
#endif
  }
  else
  {
    shmem[threadIdx.x] = (indx<r_num) ? r_x[indx] : r_x[0];                 //xmax
    shmem[threadIdx.x+blockDim.x] = (indx<r_num) ? r_x[indx+2+blockDim.x] : r_x[0]; //xmin
#ifdef _DEBUG
//    printf("2-blkid=%d, thrdid=%d, indx=%d, r_num=%d, sm_xmax=%f, sm_xmin=%f, sm_ymax=%f, sm_ymin=%f\n", blockIdx.x, threadIdx.x, indx, r_num, shmem[threadIdx.x], shmem[threadIdx.x+blockDim.x], shmem[threadIdx.x+2*blockDim.x], shmem[threadIdx.x+3*blockDim.x]);
//    printf("2-blkid=%d, thrdid=%d, indx=%d, r_num=%d, xmax=%f, xmin=%f, ymax=%f, ymin=%f, blkdim=%d\n", blockIdx.x, threadIdx.x, indx, r_num, r_x[indx], r_x[indx+2+blockDim.x], r_y[indx], r_y[indx+2+blockDim.x], blockDim.x);
#endif
  }
  __syncthreads();
  
  for(int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      if(!isnan(shmem[threadIdx.x + offset])) // xmax
      {
        if(isnan(shmem[threadIdx.x]) || shmem[threadIdx.x] < 
           shmem[threadIdx.x + offset])
          shmem[threadIdx.x] = shmem[threadIdx.x + offset];
      }
      if(!isnan(shmem[threadIdx.x+blockDim.x+offset])) // xmin
      {
        if(isnan(shmem[threadIdx.x+blockDim.x]) || shmem[threadIdx.x+blockDim.x] > 
           shmem[threadIdx.x+blockDim.x+offset])
          shmem[threadIdx.x+blockDim.x] = shmem[threadIdx.x+blockDim.x+offset];
      }
     }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!flag) // first time
    {
      r_partial_x[blockIdx.x] = shmem[0];                                // xmax
      r_partial_x[blockIdx.x+2+gridDim.x] = shmem[blockDim.x];          // xmin
#ifdef _DEBUG
//      printf("----------xmax(%d)=%f, xmin(%d)=%f, ymax(%d)=%f, ymin(%d)=%f----------------\n", 
//      blockIdx.x, r_partial_x[blockIdx.x], blockIdx.x+2+gridDim.x, r_partial_x[blockIdx.x+2+gridDim.x], 
//      blockIdx.x, r_partial_y[blockIdx.x], blockIdx.x+2+gridDim.x, r_partial_y[blockIdx.x+2+gridDim.x]);
//             shmem[0], shmem[blockDim.x], shmem[2*blockDim.x], shmem[3*blockDim.x]);
#endif
    }
    else
    {
      r_partial_x[blockIdx.x] = shmem[0];                //xmax
      r_partial_x[blockIdx.x+1] = shmem[blockDim.x];     //xmin
#ifdef _DEBUG
//      printf("2----------xmax=%f, xmin=%f, ymax=%f, ymin=%f----------------\n", 
//             shmem[0], shmem[blockDim.x], shmem[2*blockDim.x], shmem[3*blockDim.x]);
#endif
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
  {
    shmem1[threadIdx.x] = 0;
//    printf("block[%d], shmem1[%d]=%d, range=[%f, %f], r_bin_num=%d\n", blockIdx.x, threadIdx.x, shmem1[threadIdx.x],r_min, r_max, r_bin_num);
  }
  __syncthreads();
  if(indx < r_part_num && r_input_loss[indx] == 0)
  {
    volatile double x = r_x[indx];
    volatile double bin_width = (r_max-r_min)/r_bin_num;
    volatile int bin_indx = (uint)floorf((x-r_min)/bin_width);
//    printf("-data[%d] %f, before put into bin %d, r_min=%f, r_max=%f, part_num=%d\n", indx, x, bin_indx, r_min, r_max, r_part_num);
    if(bin_indx >= 0 && bin_indx  < r_bin_num-1)
    {
      atomicAdd(shmem1+bin_indx, 1);
//      printf("+data[%d] %f, put into bin %d, shmem[%d]=%d\n", indx, x, bin_indx, bin_indx, shmem1[bin_indx]);
    }
    else if(bin_indx < 0)
    {
      atomicAdd(shmem1, 1);
//      printf("+data[%d] %f, put into bin %d, shmem[%d]=%d\n", indx, x, 0, 0, shmem1[0]);
    }
    else if(bin_indx > r_bin_num-1)
    {
      atomicAdd(shmem1+r_bin_num-1, 1);
//      printf("+data[%d] %f, put into bin %d, shmem[%d]=%d\n", indx, x, r_bin_num-1, r_bin_num-1, shmem1[r_bin_num-1]);
    } 
//    printf("block[%d], data %f, put into bin %d, now shmem1[%d] = %d\n", blockIdx.x, x, bin_indx, bin_indx, shmem1[bin_indx]);
  }
  __syncthreads();
  if(threadIdx.x < r_bin_num)
  {
    r_partial[blockIdx.x*r_bin_num+threadIdx.x] = shmem1[threadIdx.x];
//    printf("partial[%d] = %d\n", blockIdx.x*r_bin_num+threadIdx.x, shmem1[threadIdx.x]); 
//    printf("partial[%d] = %d\n", blockIdx.x*r_bin_num+threadIdx.x, r_partial[blockIdx.x*r_bin_num+threadIdx.x]); 
  }
}

__global__
void HistogramReduceKernel(uint* r_partial, uint* r_hist, uint r_bin_num)
{
  extern __shared__ uint shmem2[];
  shmem2[threadIdx.x] = r_partial[threadIdx.x*r_bin_num+blockIdx.x];
  __syncthreads();
  for(int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
      shmem2[threadIdx.x] += shmem2[threadIdx.x+offset];
    __syncthreads();
  }
  if(threadIdx.x == 0)
    r_hist[blockIdx.x] = shmem2[0];
}

#include "constant.h"
__global__
void Set3dDataKernel(double* r_x, double* r_y, double* r_z, float4* r_plot_data, uint r_num)
{
  volatile uint indx = blockIdx.x*blockDim.x + threadIdx.x;
  if(indx < r_num)
  {
    double x = r_x[indx];
    double y = r_y[indx];
    double z = r_z[indx]*0.005;
    r_plot_data[indx] = make_float4((float)x, (float)y, (float)z, 0.01);
  }
}

__global__
void Histogram2DKernel(uint* r_hist, double* r_x, double* r_y, uint* r_loss, uint r_part_num,
  double r_xmin, double r_xmax, double r_ymin, double r_ymax, uint r_bin_num_x, uint r_bin_num_y)
{
  uint np = threadIdx.x + blockIdx.x*blockDim.x;  
  uint stride = blockDim.x * gridDim.x;
  while(np < r_part_num)
  {
    double x = r_x[np];
    double y = r_y[np];
    if (r_loss[np] == 0 && x >= r_xmin && x <= r_xmax && y >= r_ymin && y <= r_ymax)
    {
      double dx = (r_xmax - r_xmin)/r_bin_num_x;
      double dy = (r_ymax - r_ymin)/r_bin_num_y;
      uint indx_x = (uint) floorf((x - r_xmin)/dx);
      uint indx_y = (uint) floorf((y - r_ymin)/dy);
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
void SetHistogram2DCoordinateDataKernel(float2* r_data, double r_xmin, double r_xmax, double r_ymin, 
  double r_ymax, uint r_bin_num_x, uint r_bin_num_y)
{
  uint tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < r_bin_num_x * r_bin_num_y)
  {
    uint idy = tid / r_bin_num_x;
    uint idx = tid % r_bin_num_x;
    uint indx = idy * r_bin_num_x + idx;
    double dx = (r_xmax - r_xmin)/r_bin_num_x;
    double dy = (r_ymax - r_ymin)/r_bin_num_y;
    double x = r_xmin + dx * (0.5 + idx);
    double y = r_ymin + dy * (0.5 + idy);
    r_data[indx] = make_float2((float)x, (float)y);
  }
}
#endif
