/*! 
 * \file beam_kernel.cu
 * \brief Kernel functions for the Beam Class
 *
 * All the parallel reduction kernels of the beam class uses the algorithm
 * from <br>
 * Mark Harris, "Optimizing Parallel Reduction in CUDA", NIVDIA, 2007 <br>
 */

#ifndef BEAM_KERNEL_CU
#define BEAM_KERNEL_CU
#include "constant.h"
#include <cstdio>

/*!
 * \brief Utility class used to avoid linker errors with extern
 * unsized shared memory arrays with templated type
 */
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

/*!
 * \brief Specialized for double to avoid unaligned memory
 * access compile errors
 */
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
 * \brief Using warp-synchronous programming 
 *
 * We need to declare our shared memory volatile so that the compiler
 * doesn't reorder stores to it and induce incorrect behavior.
 */
template<class T, uint block_size>
__device__ void WarpReduce(volatile T* r_sdata, uint r_tid)
{
  if(block_size >= 64) r_sdata[r_tid] += r_sdata[r_tid + 32]; 
  if(block_size >= 32) r_sdata[r_tid] += r_sdata[r_tid + 16]; 
  if(block_size >= 16) r_sdata[r_tid] += r_sdata[r_tid +  8]; 
  if(block_size >=  8) r_sdata[r_tid] += r_sdata[r_tid +  4]; 
  if(block_size >=  4) r_sdata[r_tid] += r_sdata[r_tid +  2]; 
  if(block_size >=  2) r_sdata[r_tid] += r_sdata[r_tid +  1]; 
}

/*!
 * \brief Calculate the sum of a beam coordinate
 *
 * This version adds multiple elements per thread sequentially.  
 * This reduces the overall cost of the algorithm while keeping the work 
 * complexity O(n) and the step complexity O(log n). 
 * (Brent's Theorem optimization)
 *
 * Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
 * In other words if block_size <= 32, allocate 64*sizeof(T) bytes.
 * If block_size > 32, allocate block_size*sizeof(T) bytes.
 */
template <class T, uint block_size>
__global__ 
void XReduceKernel(T *r_idata, T *r_odata, uint r_sz, uint* r_loss, 
                   uint r_first_pass_flag, uint* r_lloss)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;

    T my_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    double tmpdata;
    uint tmploss;
    while (i < r_sz)
    {
        if(r_first_pass_flag)
        {
          tmpdata = r_idata[i];
          tmploss = r_loss[i];
          if(r_lloss != NULL)
            tmploss += r_lloss[i];
          my_sum += (tmploss == 0 ? tmpdata : 0.0);
        }
        else
          my_sum += r_idata[i];

        // ensure we don't read out of bounds -- this is optimized for powerOf2 
	// sized arrays
        if (i + block_size < r_sz)
          if(r_first_pass_flag)
          {
            tmpdata = r_idata[i + block_size];
            tmploss = r_loss[i + block_size];
            if(r_lloss != NULL)
              tmploss += r_lloss[i + block_size];
            my_sum += (tmploss == 0 ? tmpdata : 0.0);
          }
          else
            my_sum += r_idata[i + block_size];

        i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = my_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if (block_size >= 128){
        if (tid <  64)
            sdata[tid] += sdata[tid +  64];
        __syncthreads();
    }

    if (tid < 32)
      WarpReduce<T, block_size>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0)
        r_odata[blockIdx.x] = sdata[0];
}

/*!
 * \brief Kernel function that counts the number of particles which 
 *        are not transversely or longitudinally lost.
 *
 * \param r_idata Transverse loss coordinates
 * \param r_lloss Longitudinal loss coordiantes
 * \param r_odata[out] Output result
 * \param r_sz    Number of Particles
 * \param r_first_pass_flag The first time this routine is called, it reads in
 *                          the transverse and longitudinal loss coordiantes,
 *                          it doesn't for the second entry.
 */
template<uint block_size>
__global__ 
void GoodParticleCountReduceKernel(uint* r_idata, uint* r_lloss, uint* r_odata, 
    uint r_sz, uint r_first_pass_flag)
{
    uint *sdata = SharedMemory<uint>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;

    uint my_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    uint tmploss, tmplloss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmploss = r_idata[i];
        tmplloss = r_lloss[i];
        my_sum += (tmploss + tmplloss == 0 ? 1 : 0);
      }
      else 
        my_sum += r_idata[i];

      // ensure we don't read out of bounds -- this is optimized away for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
      {
        if(r_first_pass_flag)
        {
          tmploss = r_idata[i + block_size];
          tmplloss = r_lloss[i + block_size];
          my_sum += (tmploss + tmplloss == 0 ? 1 : 0);
        }
        else
          my_sum += r_idata[i + block_size];
      }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = my_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid <  64)
            sdata[tid] += sdata[tid +  64];
        __syncthreads();
    }

    if (tid < 32)
      WarpReduce<uint, block_size>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0)
        r_odata[blockIdx.x] = sdata[0];
}

/*!
 * \brief Kernel function that counts the number of non-zero elements of an
 *        uint array.
 * \param r_idata Input uint array
 * \param r_odata[out] Output result
 * \param r_sz Array size
 * \param r_first_pass_flag If true, reads from r_idata array, otherwise, 
 * 	  skip it.
 */
template<uint block_size>
__global__ 
void LossReduceKernel(uint *r_idata, uint *r_odata, uint r_sz, 
                      uint r_first_pass_flag)
{
    uint *sdata = SharedMemory<uint>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;

    uint my_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    uint tmploss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmploss = r_idata[i];
        my_sum += (tmploss == 0 ? 0 : 1);
      }
      else 
        my_sum += r_idata[i];

      // ensure we don't read out of bounds -- this is optimized away for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
      {
        if(r_first_pass_flag)
        {
          tmploss = r_idata[i + block_size];
          my_sum += (tmploss == 0 ? 0 : 1);
        }
        else
          my_sum += r_idata[i + block_size];
      }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = my_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }

    if (block_size >= 256)
    {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if (block_size >= 128)
    {
        if (tid <  64)
            sdata[tid] += sdata[tid +  64];
        __syncthreads();
    }

    if (tid < 32)
      WarpReduce<uint, block_size>(sdata, tid);

    // write result for this block to global mem
    if (tid == 0)
        r_odata[blockIdx.x] = sdata[0];
}

template<class T, uint block_size>
__device__ void WarpReduce2(volatile T* r_sdata, uint r_tid, uint r_stride)
{
  if(block_size >= 64) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 32]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 32]; 
  }
  if(block_size >= 32) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 16]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 16]; 
  }
  if(block_size >= 16) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 8]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 8]; 
  }
  if(block_size >= 8) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 4]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 4]; 
  }
  if(block_size >= 4) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 2]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 2]; 
  }
  if(block_size >= 2) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 1]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 1]; 
  }
}

template <class T, uint block_size>
__global__ 
void XX2ReduceKernel(T *r_idata, T *r_odata, uint r_sz, uint* r_loss, 
  uint r_first_pass_flag, uint* r_lloss)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;
    uint sstride = 64;
    if(block_size >= 64)
      sstride = block_size;

    T x_sum = 0;
    T x2_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    double tmpdata;
    uint tmploss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmpdata = r_idata[i];
        tmploss = r_loss[i];
        if(r_lloss != NULL)
          tmploss += r_lloss[i];
        tmpdata = (tmploss == 0 ? tmpdata : 0.0);
        x_sum += tmpdata;
        x2_sum += tmpdata * tmpdata;
      }
      else
      {
        x_sum += r_idata[i];
	// in second pass, r_sz = grid_size of the first pass
        x2_sum += r_idata[i + r_sz];  
      }

      // ensure we don't read out of bounds -- this is optimized away for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpdata = r_idata[i + block_size];
          tmploss = r_loss[i + block_size];
          if(r_lloss != NULL)
            tmploss += r_lloss[i + block_size];
          tmpdata = (tmploss == 0 ? tmpdata : 0.0);
          x_sum += tmpdata;
          x2_sum += tmpdata * tmpdata;
        }
        else
        {
          x_sum += r_idata[i + block_size];
          x2_sum += r_idata[i + block_size + r_sz]; 
        }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = x_sum;
    sdata[tid + sstride] = x2_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
        sdata[tid + sstride] += sdata[tid + sstride + 256];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
        sdata[tid + sstride] += sdata[tid + sstride + 128];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
        sdata[tid + sstride] += sdata[tid + sstride + 64];
      }
      __syncthreads();
    }

    if (tid < 32)
      WarpReduce2<T, block_size>(sdata, tid, sstride);

    // write result for this block to global mem
    if (tid == 0)
    {
      r_odata[blockIdx.x] = sdata[0];
      r_odata[blockIdx.x + gridDim.x] = sdata[sstride];
    }
}

template <class T, uint block_size>
__global__ 
void RR2ReduceKernel(T *r_idatax, T* r_idatay, T *r_odata, uint r_sz, 
  uint* r_loss, uint r_first_pass_flag)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;
    uint sstride = 64;
    if(block_size >= 64)
      sstride = block_size;

    T r_sum = 0;
    T r2_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    double tmpx = 0.0, tmpy = 0.0, tmpr2 = 0.0;
    uint tmploss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmpx = r_idatax[i];
        tmpy = r_idatay[i];
        tmploss = r_loss[i];
        if(tmploss == 0)
        {
          tmpr2 = tmpx * tmpx + tmpy * tmpy; 
          r_sum += sqrt(tmpr2);
          r2_sum += tmpr2;
        }
      }
      else
      {
        r_sum += r_idatax[i];
        // in second pass, r_sz = grid_size of the first pass
        r2_sum += r_idatax[i + r_sz]; 
      }

      // ensure we don't read out of bounds -- this is optimized for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpx = r_idatax[i + block_size];
          tmpy = r_idatay[i + block_size];
          tmploss = r_loss[i + block_size];
          if(tmploss == 0)
          {
            tmpr2 = tmpx * tmpx + tmpy * tmpy; 
            r_sum += sqrt(tmpr2);
            r2_sum += tmpr2;
          }
        }
        else
        {
          r_sum += r_idatax[i + block_size];
          r2_sum += r_idatax[i + block_size + r_sz]; 
        }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = r_sum;
    sdata[tid + sstride] = r2_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
        sdata[tid + sstride] += sdata[tid + sstride + 256];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
        sdata[tid + sstride] += sdata[tid + sstride + 128];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
        sdata[tid + sstride] += sdata[tid + sstride + 64];
      }
      __syncthreads();
    }

    if (tid < 32)
      WarpReduce2<T, block_size>(sdata, tid, sstride);

    // write result for this block to global mem
    if (tid == 0)
    {
      r_odata[blockIdx.x] = sdata[0];
      r_odata[blockIdx.x + gridDim.x] = sdata[sstride];
    }
}

template <class T, uint block_size>
__global__ 
void XYReduceKernel(T* r_idata1, T* r_idata2, T* r_odata1, T* r_odata2, 
  uint r_sz, uint* r_loss, uint r_first_pass_flag)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;
    uint sstride = 64;
    if(block_size >= 64)
      sstride = block_size;

    T x_sum = 0;
    T y_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    double tmpdata1, tmpdata2;
    uint tmploss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmpdata1 = r_idata1[i];
        tmpdata2 = r_idata2[i];
        tmploss = r_loss[i];
        if(tmploss == 0)
        {
          x_sum += tmpdata1;
          y_sum += tmpdata2;
        }
      }
      else
      {
        x_sum += r_idata1[i];
        y_sum += r_idata2[i]; 
      }

      // ensure we don't read out of bounds -- this is optimized away for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpdata1 = r_idata1[i + block_size];
          tmpdata2 = r_idata2[i + block_size];
          tmploss = r_loss[i + block_size];
          if(tmploss == 0)
          {
            x_sum += tmpdata1;
            y_sum += tmpdata2;
          }
        }
        else
        {
          x_sum += r_idata1[i + block_size];
          y_sum += r_idata2[i + block_size]; 
        }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = x_sum;
    sdata[tid + sstride] = y_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
        sdata[tid + sstride] += sdata[tid + sstride + 256];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
        sdata[tid + sstride] += sdata[tid + sstride + 128];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
        sdata[tid + sstride] += sdata[tid + sstride + 64];
      }
      __syncthreads();
    }

    if (tid < 32)
      WarpReduce2<T, block_size>(sdata, tid, sstride);

    // write result for this block to global mem
    if (tid == 0)
    {
      r_odata1[blockIdx.x] = sdata[0];
      r_odata2[blockIdx.x] = sdata[sstride];
    }
}

template<class T, uint block_size>
__device__ void WarpReduceMax2(volatile T* r_sdata, uint r_tid, uint r_stride)
{
  if(block_size >= 64) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid + 32] ? 
		    r_sdata[r_tid + 32] : r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid + r_stride] < 
      r_sdata[r_tid + r_stride + 32] ? r_sdata[r_tid + r_stride + 32] : 
      r_sdata[r_tid + r_stride];
  }
  if(block_size >= 32) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid + 16] ? 
		    r_sdata[r_tid + 16] : r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid + r_stride] < 
      r_sdata[r_tid + r_stride + 16] ? r_sdata[r_tid + r_stride + 16] : 
      r_sdata[r_tid + r_stride];
  }
  if(block_size >= 16) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid + 8] ? r_sdata[r_tid + 8] : 
		     r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid + r_stride] < 
      r_sdata[r_tid + r_stride + 8] ? r_sdata[r_tid + r_stride + 8] : 
      r_sdata[r_tid + r_stride];
  }
  if(block_size >= 8) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid + 4] ? r_sdata[r_tid + 4] : 
		     r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid + r_stride] < 
      r_sdata[r_tid + r_stride + 4] ? r_sdata[r_tid + r_stride + 4] : 
      r_sdata[r_tid + r_stride];
  }
  if(block_size >= 4) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid + 2] ? r_sdata[r_tid + 2] : 
		     r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid + r_stride] < 
      r_sdata[r_tid + r_stride + 2] ? r_sdata[r_tid + r_stride + 2] : 
      r_sdata[r_tid + r_stride];
  }
  if(block_size >= 2) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid + 1] ? r_sdata[r_tid + 1] : 
		     r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid + r_stride] < 
      r_sdata[r_tid + r_stride + 1] ? r_sdata[r_tid + r_stride + 1] : 
      r_sdata[r_tid + r_stride];
  }
}

/*!
 * \brief Calculate max r^=x^2+y^2 and max absolute phase simultaneously
 *
 * \param r_idata1 Coordiante x
 * \param r_idata2 Coordinate y
 * \param r_idata3 Coordinate relative phase
 * \param r_loss, Coordinate transverse loss
 * \param r_x_avg Average x
 * \param r_y_avg Average y
 * \param r_odata1[out] Output, after second entry, it holds max(r^2)
 * \param r_odata2[out] Output, after second entry, it holds max(phi)
 * \param r_sz Particle number
 * \param r_first_pass_flag, If true, read from inputs, otherwise, skip it.
 */
template <class T, uint block_size>
__global__ 
void MaxR2PhiReduceKernel(T* r_idata1, T* r_idata2, T* r_idata3, uint* r_loss,
  T* r_x_avg, T* r_y_avg, T* r_phi_avg, T* r_odata1, T* r_odata2, uint r_sz, 
  uint r_first_pass_flag)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;
    uint sstride = 64;
    if(block_size >= 64)
      sstride = block_size;

    T r2_max = 0;
    T phi_max = 0;
    T x_avg;
    T y_avg;
    T phi_avg;
    if(r_first_pass_flag)
    {
      x_avg = r_x_avg[0];
      y_avg = r_y_avg[0];
      phi_avg = r_phi_avg[0];
    }

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    double tmpx, tmpy, tmpr2, tmpphi;
    uint tmploss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmpx = r_idata1[i];
        tmpy = r_idata2[i];
        tmpphi = r_idata3[i];
        tmploss = r_loss[i];
        if(tmploss == 0)
        {
          tmpx -= x_avg;
          tmpy -= y_avg;
          tmpr2 = tmpx * tmpx + tmpy * tmpy;
          tmpphi -= phi_avg;
          tmpphi = tmpphi > 0.0 ? tmpphi : -tmpphi;
          r2_max = r2_max < tmpr2 ? tmpr2 : r2_max;
          phi_max = phi_max < tmpphi ? tmpphi : phi_max; 
        }
      }
      else
      {
        tmpr2 = r_idata1[i];
        tmpphi = r_idata2[i];
        r2_max = r2_max < tmpr2 ? tmpr2 : r2_max;
        phi_max = phi_max < tmpphi ? tmpphi : phi_max; 
      }

      // ensure we don't read out of bounds -- this is optimized away for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpx = r_idata1[i + block_size];
          tmpy = r_idata2[i + block_size];
          tmpphi = r_idata3[i + block_size];
          tmploss = r_loss[i + block_size];
          if(tmploss == 0)
          {
            tmpx -= x_avg;
            tmpy -= y_avg;
            tmpr2 = tmpx * tmpx + tmpy * tmpy;
            tmpphi -= phi_avg;
            tmpphi = tmpphi > 0.0 ? tmpphi : -tmpphi;
            r2_max = r2_max < tmpr2 ? tmpr2 : r2_max;
            phi_max = phi_max < tmpphi ? tmpphi : phi_max; 
          }
        }
        else
        {
          tmpr2 = r_idata1[i + block_size];
          tmpphi = r_idata2[i + block_size];
          r2_max = r2_max < tmpr2 ? tmpr2 : r2_max;
          phi_max = phi_max < tmpphi ? tmpphi : phi_max;
        }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = r2_max;
    sdata[tid + sstride] = phi_max;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] = sdata[tid] < sdata[tid+256] ? sdata[tid+256] : sdata[tid];
        sdata[tid+sstride] = sdata[tid+sstride] < sdata[tid+sstride+256] ? 
                            sdata[tid+sstride+256] : sdata[tid+sstride];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] = sdata[tid] < sdata[tid+128] ? sdata[tid+128] : sdata[tid];
        sdata[tid+sstride] = sdata[tid+sstride] < sdata[tid+sstride+128] ? 
                            sdata[tid+sstride+128] : sdata[tid+sstride];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] = sdata[tid] < sdata[tid+64] ? sdata[tid+64] : sdata[tid];
        sdata[tid+sstride] = sdata[tid+sstride] < sdata[tid+sstride+64] ? 
                            sdata[tid+sstride+64] : sdata[tid+sstride];
      }
      __syncthreads();
    }

    if (tid < 32)
      WarpReduceMax2<T, block_size>(sdata, tid, sstride);

    // write result for this block to global mem
    if (tid == 0)
    {
      r_odata1[blockIdx.x] = sdata[0];
      r_odata2[blockIdx.x] = sdata[sstride];
    }
}

template<class T, uint block_size>
__device__ void WarpReduce4(volatile T* r_sdata, uint r_tid, uint r_stride)
{
  if(block_size >= 64) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 32]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 32]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 32]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 32]; 
  }
  if(block_size >= 32) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 16]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 16]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 16]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 16]; 
  }
  if(block_size >= 16) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 8]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 8]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 8]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 8]; 
  }
  if(block_size >= 8) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 4]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 4]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 4]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 4]; 
  }
  if(block_size >= 4) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 2]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 2]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 2]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 2]; 
  }
  if(block_size >= 2) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 1]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 1]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 1]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 1]; 
  }
}

template <class T, uint block_size>
__global__ 
void XYX2Y2ReduceKernel(T* r_idata1, T* r_idata2, T* r_odata1, T* r_odata2, 
  uint r_sz, uint* r_loss, uint* r_lloss, uint r_first_pass_flag)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;
    uint sstride = 64;
    if(block_size >= 64)
      sstride = block_size;

    T x_sum = 0;
    T x2_sum = 0;
    T y_sum = 0;
    T y2_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    double tmpdata1, tmpdata2;
    uint tmploss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmpdata1 = r_idata1[i];
        tmpdata2 = r_idata2[i];
        tmploss = r_loss[i];
        tmploss += r_lloss[i];
        if(tmploss == 0)
        {
          x_sum += tmpdata1;
          x2_sum += tmpdata1 * tmpdata1;
          y_sum += tmpdata2;
          y2_sum += tmpdata2 * tmpdata2;
        }
      }
      else
      {
        x_sum += r_idata1[i];
	// r_sz in second pass = grid_size of the first pass
        x2_sum += r_idata1[i + r_sz]; 
        y_sum += r_idata2[i]; 
        y2_sum += r_idata2[i + r_sz];
      }

      // ensure we don't read out of bounds -- this is optimized away for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpdata1 = r_idata1[i + block_size];
          tmpdata2 = r_idata2[i + block_size];
          tmploss = r_loss[i + block_size];
          tmploss += r_lloss[i + block_size];
          if(tmploss == 0)
          {
            x_sum += tmpdata1;
            x2_sum += tmpdata1 * tmpdata1;
            y_sum += tmpdata2;
            y2_sum += tmpdata2 * tmpdata2;
          }
        }
        else
        {
          x_sum += r_idata1[i + block_size];
          x2_sum += r_idata1[i + r_sz + block_size];
          y_sum += r_idata2[i + block_size]; 
          y2_sum += r_idata2[i + r_sz + block_size]; 
        }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = x_sum;
    sdata[tid + sstride] = x2_sum;
    sdata[tid + 2 * sstride] = y_sum;
    sdata[tid + 3 * sstride] = y2_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
        sdata[tid + sstride] += sdata[tid + sstride + 256];
        sdata[tid + 2 * sstride] += sdata[tid + 2 * sstride + 256];
        sdata[tid + 3 * sstride] += sdata[tid + 3 * sstride + 256];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
        sdata[tid + sstride] += sdata[tid + sstride + 128];
        sdata[tid + 2 * sstride] += sdata[tid + 2 * sstride + 128];
        sdata[tid + 3 * sstride] += sdata[tid + 3 * sstride + 128];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
        sdata[tid + sstride] += sdata[tid + sstride + 64];
        sdata[tid + 2 * sstride] += sdata[tid + 2 * sstride + 64];
        sdata[tid + 3 * sstride] += sdata[tid + 3 * sstride + 64];
      }
      __syncthreads();
    }

    if (tid < 32)
      WarpReduce4<T, block_size>(sdata, tid, sstride);

    // write result for this block to global mem
    if (tid == 0)
    {
      r_odata1[blockIdx.x] = sdata[0];  // x
      r_odata1[blockIdx.x + gridDim.x] = sdata[sstride];  // x2
      r_odata2[blockIdx.x] = sdata[2 * sstride];  // y
      r_odata2[blockIdx.x + gridDim.x] = sdata[3 * sstride];// y2
    }
}

template<class T, uint block_size>
__device__ void WarpReduce5(volatile T* r_sdata, uint r_tid, uint r_stride)
{
  if(block_size >= 64) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 32]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 32]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 32]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 32]; 
    r_sdata[r_tid + 4 * r_stride] += r_sdata[r_tid + 4 * r_stride + 32]; 
  }
  if(block_size >= 32) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 16]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 16]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 16]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 16]; 
    r_sdata[r_tid + 4 * r_stride] += r_sdata[r_tid + 4 * r_stride + 16]; 
  }
  if(block_size >= 16) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 8]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 8]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 8]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 8]; 
    r_sdata[r_tid + 4 * r_stride] += r_sdata[r_tid + 4 * r_stride + 8]; 
  }
  if(block_size >= 8) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 4]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 4]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 4]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 4]; 
    r_sdata[r_tid + 4 * r_stride] += r_sdata[r_tid + 4 * r_stride + 4]; 
  }
  if(block_size >= 4) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 2]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 2]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 2]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 2]; 
    r_sdata[r_tid + 4 * r_stride] += r_sdata[r_tid + 4 * r_stride + 2]; 
  }
  if(block_size >= 2) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 1]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 1]; 
    r_sdata[r_tid + 2 * r_stride] += r_sdata[r_tid + 2 * r_stride + 1]; 
    r_sdata[r_tid + 3 * r_stride] += r_sdata[r_tid + 3 * r_stride + 1]; 
    r_sdata[r_tid + 4 * r_stride] += r_sdata[r_tid + 4 * r_stride + 1]; 
  }
}

/*!
 * \brief Kernel function that calculates the sum of x, x^2, xp, xp^2 and x*xp 
 *        of a beam coordinate simultaneously
 *
 * \param r_idata1 Coordinate x or y or relative phase
 * \param r_idata2 Coordinate xp or yp or kinetic energy
 * \param r_odata1[out] Partial result output
 * \param r_odata2[out] Partial result output
 * \param r_sz Particle number
 * \param r_loss Coordiate transverse loss
 * \param r_first_pass_flag If ture, read input data, otherwise, skip it.
 * 
 * If r_frist_pass_flag = false (second entry), then the outputs are <br>
 * r_odata1[0] stores sum of x/y/phi<br>
 * r_odata1[1] stores sum of x^2/y^2/phi^2 <br>
 * r_odata2[0] stores sum of xp/yp/w<br>
 * r_odata2[1] stores sum of xp^2/yp^2/w^2<br>
 * r_odata2[2] stores sum of x*xp/y*yp/phi*w
 */
template <class T, uint block_size>
__global__ 
void EmittanceReduceKernel(T* r_idata1, T* r_idata2, T* r_odata1, T* r_odata2, 
  uint r_sz, uint* r_loss, uint r_first_pass_flag)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    uint tid = threadIdx.x;
    uint i = blockIdx.x * block_size * 2 + threadIdx.x;
    uint grid_size = block_size * 2 * gridDim.x;
    uint sstride = 64;
    if(block_size >= 64)
      sstride = block_size;

    T x_sum = 0;
    T x2_sum = 0;
    T xp_sum = 0;
    T xp2_sum = 0;
    T xxp_sum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger grid_size and therefore fewer elements per thread
    double tmpdata1, tmpdata2;
    uint tmploss;
    while (i < r_sz)
    {
      if(r_first_pass_flag)
      {
        tmpdata1 = r_idata1[i];
        tmpdata2 = r_idata2[i];
        tmploss = r_loss[i];
        if(tmploss == 0)
        {
          x_sum += tmpdata1;
          x2_sum += tmpdata1 * tmpdata1;
          xp_sum += tmpdata2;
          xp2_sum += tmpdata2 * tmpdata2;
          xxp_sum += tmpdata1 * tmpdata2;
        }
      }
      else
      {
        x_sum += r_idata1[i];
	// r_sz in second pass = grid_size of the first pass
        x2_sum += r_idata1[i + r_sz]; 
        xp_sum += r_idata2[i]; 
        xp2_sum += r_idata2[i + r_sz];
        xxp_sum += r_idata2[i + 2 * r_sz];
      }

      // ensure we don't read out of bounds -- this is optimized away for 
      // powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpdata1 = r_idata1[i + block_size];
          tmpdata2 = r_idata2[i + block_size];
          tmploss = r_loss[i + block_size];
          if(tmploss == 0)
          {
            x_sum += tmpdata1;
            x2_sum += tmpdata1 * tmpdata1;
            xp_sum += tmpdata2;
            xp2_sum += tmpdata2 * tmpdata2;
            xxp_sum += tmpdata1 * tmpdata2;
          }
        }
        else
        {
          x_sum += r_idata1[i + block_size];
          x2_sum += r_idata1[i + r_sz + block_size];
          xp_sum += r_idata2[i + block_size]; 
          xp2_sum += r_idata2[i + r_sz + block_size]; 
          xxp_sum += r_idata2[i + 2 * r_sz + block_size]; 
        }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = x_sum;
    sdata[tid + sstride] = x2_sum;
    sdata[tid + 2 * sstride] = xp_sum;
    sdata[tid + 3 * sstride] = xp2_sum;
    sdata[tid + 4 * sstride] = xxp_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
        sdata[tid + sstride] += sdata[tid + sstride + 256];
        sdata[tid + 2 * sstride] += sdata[tid + 2 * sstride + 256];
        sdata[tid + 3 * sstride] += sdata[tid + 3 * sstride + 256];
        sdata[tid + 4 * sstride] += sdata[tid + 4 * sstride + 256];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
        sdata[tid + sstride] += sdata[tid + sstride + 128];
        sdata[tid + 2 * sstride] += sdata[tid + 2 * sstride + 128];
        sdata[tid + 3 * sstride] += sdata[tid + 3 * sstride + 128];
        sdata[tid + 4 * sstride] += sdata[tid + 4 * sstride + 128];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
        sdata[tid + sstride] += sdata[tid + sstride + 64];
        sdata[tid + 2 * sstride] += sdata[tid + 2 * sstride + 64];
        sdata[tid + 3 * sstride] += sdata[tid + 3 * sstride + 64];
        sdata[tid + 4 * sstride] += sdata[tid + 4 * sstride + 64];
      }
      __syncthreads();
    }

    if (tid < 32)
      WarpReduce5<T, block_size>(sdata, tid, sstride);

    // write result for this block to global mem
    if (tid == 0)
    {
      r_odata1[blockIdx.x] = sdata[0];  // x
      r_odata1[blockIdx.x + gridDim.x] = sdata[sstride];  // x2
      r_odata2[blockIdx.x] = sdata[2 * sstride];  // xp 
      r_odata2[blockIdx.x + gridDim.x] = sdata[3 * sstride];// xp2
      r_odata2[blockIdx.x + 2 * gridDim.x] = sdata[4 * sstride];// xxp
    }
}

__global__
void UpdateTransverseEmittanceKernel(double* r_xavg, double* r_xsig, 
  double* r_xpavg, double* r_xpsig,
  double* r_xeps, double* r_partial1, double* r_partial2, uint r_num, 
  uint* r_loss, double* r_w_ref, double r_mass)
{
  volatile double surv_num = r_num - (*r_loss);
  volatile double x = r_partial1[0] / surv_num;
  volatile double x2 = r_partial1[1] / surv_num;
  volatile double xp = r_partial2[0] / surv_num;
  volatile double xp2 = r_partial2[1] / surv_num;
  volatile double xxp = r_partial2[2] / surv_num;
  volatile double gamma1 = *r_w_ref / r_mass;
  volatile double betagamma = sqrt(gamma1 * (gamma1 + 2.0));
  r_xavg[0] = x;
  r_xsig[0] = sqrt(x2 - x * x);
  r_xpavg[0] = xp;
  r_xpsig[0] = sqrt(xp2 - xp * xp);
  //r_xeps[0] = sqrt(x2 * xp2-xxp * xxp) * betagamma;
  r_xeps[0] = sqrt((x2 - x * x) * (xp2 - xp * xp) - 
    (xxp - x * xp) * (xxp - x * xp)) * betagamma;
}

__global__
void UpdateLongitudinalEmittanceKernel(double* r_phi, double* r_phisig, 
  double* r_w, double* r_wsig, double* r_leps, double* r_partial1, 
  double* r_partial2, uint r_num, uint* r_loss)
{
  volatile double surv_num = r_num - (*r_loss);
  volatile double phi = r_partial1[0] / surv_num;
  volatile double phi2 = r_partial1[1] / surv_num;
  volatile double w = r_partial2[0] / surv_num;
  volatile double w2 = r_partial2[1] / surv_num;
  volatile double phi_w = r_partial2[2] / surv_num;
  volatile double phi_var = phi2 - phi * phi;
  volatile double w_var = w2 - w * w; 
  volatile double phi_w_covar = phi_w-phi*w;
  r_phi[0] = phi; //-*r_phi_ref;///RADIAN;
  r_w[0] = w; //-*r_w_ref;
  r_phisig[0] = sqrt(phi_var);
  r_wsig[0] = sqrt(w_var);
  r_leps[0] = sqrt(phi_var * w_var - phi_w_covar * phi_w_covar);
}

__global__
void UpdateMaxRPhiKernel(double* r_r_max, double* r_abs_phi_max, 
  double* r_r2_max_val, double* r_abs_phi_max_val)
{
  r_r_max[0] = sqrt(r_r2_max_val[0]);
  r_abs_phi_max[0] = r_abs_phi_max_val[0];
}

__global__
void UpdateHorizontalAvgXYKernel(double* r_x_avg, double* r_y_avg,
  double* r_x_avg_val, double* r_y_avg_val, uint r_num, uint* r_loss)
{
  volatile double surv_num = r_num - (*r_loss);
  volatile double xavg = r_x_avg_val[0] / surv_num;
  volatile double yavg = r_y_avg_val[0] / surv_num;
  r_x_avg[0] = xavg;
  r_y_avg[0] = yavg;
}

__global__
void UpdateHorizontalAvgSigKernel(double* r_x_avg, double* r_x_sig,
  double* r_y_avg, double* r_y_sig, double* r_x_x2, double* r_y_y2, 
  uint r_num, uint* r_loss)
{
  volatile double surv_num = r_num - (*r_loss);
  volatile double xavg = r_x_x2[0] / surv_num;
  volatile double x2avg = r_x_x2[1] / surv_num;
  volatile double yavg = r_y_y2[0] / surv_num;
  volatile double y2avg = r_y_y2[1] / surv_num;
  r_x_avg[0] = xavg;
  r_x_sig[0] = sqrt(x2avg - xavg * xavg);
  r_y_avg[0] = yavg;
  r_y_sig[0] = sqrt(y2avg - yavg * yavg);
}

__global__
void UpdateVariableAvgKernel(double* r_avg, double* r_sum,
                                    uint r_num, uint* r_loss)
{
  double surv_num = r_num - (*r_loss);
  r_avg[0] = r_sum[0] / surv_num;
}

__global__
void UpdateVariableAvgWithLlossKernel(double* r_avg, double* r_sum, uint* r_num)
{
  if(r_num[0] > 0)
    r_avg[0] = r_sum[0] / r_num[0];
  else
    r_avg[0] = -99999;
}

__global__
void UpdateVariableSigmaKernel(double* r_sig, double* r_sum,
                           double* r_sum2, uint r_num, uint* r_loss)
{
  double surv_num = r_num - (*r_loss);
  double avg = r_sum[0] / surv_num;
  double avg2 = r_sum2[0] / surv_num;
  r_sig[0] = sqrt(avg2 - avg * avg);
}

__global__
void UpdateVariableSigmaWithLlossKernel(double* r_sig, double* r_sum,
                           double* r_sum2, uint* r_num)
{
  double avg = r_sum[0] / r_num[0];
  double avg2 = r_sum2[0] / r_num[0];
  r_sig[0] = sqrt(avg2 - avg  *  avg);
}

template<typename T>
__global__
void SetValueKernel(T* r_pos, uint r_index, T r_val)
{
  r_pos[r_index] = r_val;
}

/*!
 * \brief Kernel function that cuts beam
 *
 * \param r_x Coordinate which the cut will be applied to
 * \param r_loss Transverse loss coordinates
 * \param r_min Cut lower range
 * \param r_max Cut upper range
 * \param r_num Particle number 
 */
__global__
void CutBeamKernel(double* r_x, uint* r_loss, double r_min, double r_max, 
  uint r_num)
{
  uint index = blockIdx.x * blockDim.x+threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  while(index > 0 && index < r_num)
  {
    if(r_loss[index] == 0)
    {
      double x = r_x[index];
      if(x < r_max &&  x > r_min)
        r_loss[index] = 99999;
    }
    index += stride;
  }
}

/*!
 * \brief Kernel function that shifts beam coorindates
 *
 * \param r_var Coordiante to be shifted
 * \param r_val Shift amount
 * \param r_sz Particle Number
 */
template<typename T>
__global__
void ShiftVariableKernel(T* r_var, T r_val, uint r_sz)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > 0 && index < r_sz)
    r_var[index] += r_val;
}

/*!
 * \brief Kernel function that updates the relative phase coordinates
 *
 * \param r_phi_r[out] Coordinate relative phase
 * \param r_phi Coordinate absolute phase
 * \param r_ref_phi The reference phase
 * \param r_sz Particle number
 */
__global__
void UpdateRelativePhiKernel(double* r_phi_r, double* r_phi, double* r_ref_phi, 
  uint r_sz)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;  
  if(index < r_sz)
  {
    double tmp = r_phi[index] - r_ref_phi[0] + PI;
    double sign = 1.0;
    if (tmp < 0)
    {
      sign = -1.0;
      tmp = -tmp;
    }
    r_phi_r[index] = sign * (fmod(tmp, TWOPI) - PI);
  }     
} 

/*!
 * \brief Kernel function to change beam phase after frequency change.
 *
 * \param r_phi Coordinate absolute phase
 * \param r_freq_ratio Ratio of new frequency over the old one 
 * \param r_sz Particle number
 */
__global__
void ChangeFrequencyKernel(double* r_phi, double r_freq_ratio, uint r_sz)
{
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index < r_sz)
    r_phi[index] *= r_freq_ratio;
}

/*!
 * \brief Kernel function that updates the longitudinal loss coordinates 
 *
 * \param r_lloss[out] Longitudinal loss coordinates
 * \param r_phi Absolute phase coordinates
 * \param r_phi_avg Average of the absolute phase
 * \param r_sz Particle number
 * A particle is considered to be lost longitudinally if its absolute phase
 * is 3*PI away from the average.
 */
__global__
void UpdateLongitudinalLossCoordinateKernel(uint* r_lloss, double* r_phi, 
  double* r_phi_avg, uint r_sz)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < r_sz)
  {
    if(r_phi[index] > r_phi_avg[0] + TWOPI + PI)
      r_lloss[index] = 1;
  }
}
#endif
