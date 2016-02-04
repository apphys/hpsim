#ifndef BEAM_KERNEL_CU
#define BEAM_KERNEL_CU
#include "constant.h"
#include <cstdio>

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
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

// specialize for double to avoid unaligned memory
// access compile errors
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

// now that we are using warp-synchronous programming (below)
// we need to declare our shared memory volatile so that the compiler
// doesn't reorder stores to it and induce incorrect behavior.
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

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if block_size <= 32, allocate 64*sizeof(T) bytes.
    If block_size > 32, allocate block_size*sizeof(T) bytes.
*/
template <class T, uint block_size>
__global__ 
void XReduceKernel(T *r_idata, T *r_odata, uint r_sz, uint* r_loss, 
                   uint r_first_pass_flag, uint* r_lloss)
{
//    printf("XReduceKernel: lloss ? %d\n", (r_lloss != NULL));
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

        // ensure we don't read out of bounds -- this is optimized for powerOf2 sized arrays
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

      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
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

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
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
        x2_sum += tmpdata*tmpdata;
      }
      else
      {
        x_sum += r_idata[i];
        x2_sum += r_idata[i + r_sz];  // in second pass, r_sz = grid_size of the first pass
      }

      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpdata = r_idata[i + block_size];
          tmploss = r_loss[i + block_size];
          if(r_lloss != NULL)
            tmploss += r_lloss[i + block_size];
          tmpdata = (tmploss == 0 ? tmpdata : 0.0);
          x_sum += tmpdata;
          x2_sum += tmpdata*tmpdata;
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
void RR2ReduceKernel(T *r_idatax, T* r_idatay, T *r_odata, uint r_sz, uint* r_loss, uint r_first_pass_flag)
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
          tmpr2 = tmpx*tmpx + tmpy*tmpy; 
          r_sum += sqrt(tmpr2);
          r2_sum += tmpr2;
        }
      }
      else
      {
        r_sum += r_idatax[i];
        r2_sum += r_idatax[i + r_sz];  // in second pass, r_sz = grid_size of the first pass
      }

      // ensure we don't read out of bounds -- this is optimized for powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpx = r_idatax[i + block_size];
          tmpy = r_idatay[i + block_size];
          tmploss = r_loss[i + block_size];
          if(tmploss == 0)
          {
            tmpr2 = tmpx*tmpx + tmpy*tmpy; 
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
void XYReduceKernel(T* r_idata1, T* r_idata2, T* r_odata1, T* r_odata2, uint r_sz, uint* r_loss, uint r_first_pass_flag)
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

      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
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
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid+32] ? r_sdata[r_tid+32] : 
                      r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid+r_stride] < r_sdata[r_tid+r_stride+32] ? 
                        r_sdata[r_tid+r_stride+32] : r_sdata[r_tid+r_stride];
  }
  if(block_size >= 32) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid+16] ? r_sdata[r_tid+16] : r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid+r_stride] < r_sdata[r_tid+r_stride+16] ? 
                        r_sdata[r_tid+r_stride+16] : r_sdata[r_tid+r_stride];
  }
  if(block_size >= 16) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid+8] ? r_sdata[r_tid+8] : r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid+r_stride] < r_sdata[r_tid+r_stride+8] ? 
                        r_sdata[r_tid+r_stride+8] : r_sdata[r_tid+r_stride];
  }
  if(block_size >= 8) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid+4] ? r_sdata[r_tid+4] : r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid+r_stride] < r_sdata[r_tid+r_stride+4] ? 
                        r_sdata[r_tid+r_stride+4] : r_sdata[r_tid+r_stride];
  }
  if(block_size >= 4) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid+2] ? r_sdata[r_tid+2] : r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid+r_stride] < r_sdata[r_tid+r_stride+2] ? 
                        r_sdata[r_tid+r_stride+2] : r_sdata[r_tid+r_stride];
  }
  if(block_size >= 2) 
  {
    r_sdata[r_tid] = r_sdata[r_tid] < r_sdata[r_tid+1] ? r_sdata[r_tid+1] : r_sdata[r_tid];
    r_sdata[r_tid+r_stride] = r_sdata[r_tid+r_stride] < r_sdata[r_tid+r_stride+1] ? 
                        r_sdata[r_tid+r_stride+1] : r_sdata[r_tid+r_stride];
  }
}

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
          tmpr2 = tmpx*tmpx + tmpy*tmpy;
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
//        if(!r_first_pass_flag)
//          printf("tid=%d, r2_max=%15.13f, phi_max=%15.13f\n", tid, r2_max, phi_max);
      }

      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
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
            tmpr2 = tmpx*tmpx + tmpy*tmpy;
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
//          if(!r_first_pass_flag)
//            printf("tid=%d, r2_max=%15.13f, phi_max=%15.13f\n", tid, r2_max, phi_max);
        }
      i += grid_size;
    }

//    if(!r_first_pass_flag)
//      printf("||tid=%d, r2_max=%15.13f, phi_max=%15.13f\n", tid, r2_max, phi_max);
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
//      printf("!!!max kernel: r2_max = %15.13f, phi_max = %15.13f\n", sdata[0], sdata[sstride]);
    }
}

template<class T, uint block_size>
__device__ void WarpReduce4(volatile T* r_sdata, uint r_tid, uint r_stride)
{
  if(block_size >= 64) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 32]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 32]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 32]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 32]; 
  }
  if(block_size >= 32) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 16]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 16]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 16]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 16]; 
  }
  if(block_size >= 16) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 8]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 8]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 8]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 8]; 
  }
  if(block_size >= 8) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 4]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 4]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 4]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 4]; 
  }
  if(block_size >= 4) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 2]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 2]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 2]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 2]; 
  }
  if(block_size >= 2) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 1]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 1]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 1]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 1]; 
  }
}

template <class T, uint block_size>
__global__ 
void XYX2Y2ReduceKernel(T* r_idata1, T* r_idata2, T* r_odata1, T* r_odata2, uint r_sz, 
                        uint* r_loss, uint* r_lloss, uint r_first_pass_flag)
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
          x2_sum += tmpdata1*tmpdata1;
          y_sum += tmpdata2;
          y2_sum += tmpdata2*tmpdata2;
        }
      }
      else
      {
        x_sum += r_idata1[i];
        x2_sum += r_idata1[i + r_sz]; // r_sz in second pass = grid_size of the first pass
        y_sum += r_idata2[i]; 
        y2_sum += r_idata2[i + r_sz];
      }

      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
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
            x2_sum += tmpdata1*tmpdata1;
            y_sum += tmpdata2;
            y2_sum += tmpdata2*tmpdata2;
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
    sdata[tid + 2*sstride] = y_sum;
    sdata[tid + 3*sstride] = y2_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
        sdata[tid + sstride] += sdata[tid + sstride + 256];
        sdata[tid + 2*sstride] += sdata[tid + 2*sstride + 256];
        sdata[tid + 3*sstride] += sdata[tid + 3*sstride + 256];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
        sdata[tid + sstride] += sdata[tid + sstride + 128];
        sdata[tid + 2*sstride] += sdata[tid + 2*sstride + 128];
        sdata[tid + 3*sstride] += sdata[tid + 3*sstride + 128];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
        sdata[tid + sstride] += sdata[tid + sstride + 64];
        sdata[tid + 2*sstride] += sdata[tid + 2*sstride + 64];
        sdata[tid + 3*sstride] += sdata[tid + 3*sstride + 64];
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
      r_odata2[blockIdx.x] = sdata[2*sstride];  // y
      r_odata2[blockIdx.x + gridDim.x] = sdata[3*sstride];// y2
    }
}

template<class T, uint block_size>
__device__ void WarpReduce5(volatile T* r_sdata, uint r_tid, uint r_stride)
{
  if(block_size >= 64) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 32]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 32]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 32]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 32]; 
    r_sdata[r_tid + 4*r_stride] += r_sdata[r_tid + 4*r_stride + 32]; 
  }
  if(block_size >= 32) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 16]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 16]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 16]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 16]; 
    r_sdata[r_tid + 4*r_stride] += r_sdata[r_tid + 4*r_stride + 16]; 
  }
  if(block_size >= 16) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 8]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 8]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 8]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 8]; 
    r_sdata[r_tid + 4*r_stride] += r_sdata[r_tid + 4*r_stride + 8]; 
  }
  if(block_size >= 8) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 4]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 4]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 4]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 4]; 
    r_sdata[r_tid + 4*r_stride] += r_sdata[r_tid + 4*r_stride + 4]; 
  }
  if(block_size >= 4) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 2]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 2]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 2]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 2]; 
    r_sdata[r_tid + 4*r_stride] += r_sdata[r_tid + 4*r_stride + 2]; 
  }
  if(block_size >= 2) 
  {
    r_sdata[r_tid] += r_sdata[r_tid + 1]; 
    r_sdata[r_tid + r_stride] += r_sdata[r_tid + r_stride + 1]; 
    r_sdata[r_tid + 2*r_stride] += r_sdata[r_tid + 2*r_stride + 1]; 
    r_sdata[r_tid + 3*r_stride] += r_sdata[r_tid + 3*r_stride + 1]; 
    r_sdata[r_tid + 4*r_stride] += r_sdata[r_tid + 4*r_stride + 1]; 
  }
}

template <class T, uint block_size>
__global__ 
void EmittanceReduceKernel(T* r_idata1, T* r_idata2, T* r_odata1, T* r_odata2, uint r_sz, uint* r_loss, uint r_first_pass_flag)
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
          x2_sum += tmpdata1*tmpdata1;
          xp_sum += tmpdata2;
          xp2_sum += tmpdata2*tmpdata2;
          xxp_sum += tmpdata1*tmpdata2;
        }
      }
      else
      {
        x_sum += r_idata1[i];
        x2_sum += r_idata1[i + r_sz]; // r_sz in second pass = grid_size of the first pass
        xp_sum += r_idata2[i]; 
        xp2_sum += r_idata2[i + r_sz];
        xxp_sum += r_idata2[i + 2*r_sz];
      }

      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if (i + block_size < r_sz)
        if(r_first_pass_flag)
        {
          tmpdata1 = r_idata1[i + block_size];
          tmpdata2 = r_idata2[i + block_size];
          tmploss = r_loss[i + block_size];
          if(tmploss == 0)
          {
            x_sum += tmpdata1;
            x2_sum += tmpdata1*tmpdata1;
            xp_sum += tmpdata2;
            xp2_sum += tmpdata2*tmpdata2;
            xxp_sum += tmpdata1*tmpdata2;
          }
        }
        else
        {
          x_sum += r_idata1[i + block_size];
          x2_sum += r_idata1[i + r_sz + block_size];
          xp_sum += r_idata2[i + block_size]; 
          xp2_sum += r_idata2[i + r_sz + block_size]; 
          xxp_sum += r_idata2[i + 2*r_sz + block_size]; 
        }
      i += grid_size;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = x_sum;
    sdata[tid + sstride] = x2_sum;
    sdata[tid + 2*sstride] = xp_sum;
    sdata[tid + 3*sstride] = xp2_sum;
    sdata[tid + 4*sstride] = xxp_sum;
    __syncthreads();


    // do reduction in shared mem
    if (block_size >= 512)
    {
      if (tid < 256)
      {
        sdata[tid] += sdata[tid + 256];
        sdata[tid + sstride] += sdata[tid + sstride + 256];
        sdata[tid + 2*sstride] += sdata[tid + 2*sstride + 256];
        sdata[tid + 3*sstride] += sdata[tid + 3*sstride + 256];
        sdata[tid + 4*sstride] += sdata[tid + 4*sstride + 256];
      }
      __syncthreads();
    }

    if (block_size >= 256)
    {
      if (tid < 128)
      {
        sdata[tid] += sdata[tid + 128];
        sdata[tid + sstride] += sdata[tid + sstride + 128];
        sdata[tid + 2*sstride] += sdata[tid + 2*sstride + 128];
        sdata[tid + 3*sstride] += sdata[tid + 3*sstride + 128];
        sdata[tid + 4*sstride] += sdata[tid + 4*sstride + 128];
      }
      __syncthreads();
    }

    if (block_size >= 128)
    {
      if (tid <  64)
      {
        sdata[tid] += sdata[tid +  64];
        sdata[tid + sstride] += sdata[tid + sstride + 64];
        sdata[tid + 2*sstride] += sdata[tid + 2*sstride + 64];
        sdata[tid + 3*sstride] += sdata[tid + 3*sstride + 64];
        sdata[tid + 4*sstride] += sdata[tid + 4*sstride + 64];
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
      r_odata2[blockIdx.x] = sdata[2*sstride];  // xp 
      r_odata2[blockIdx.x + gridDim.x] = sdata[3*sstride];// xp2
      r_odata2[blockIdx.x + 2*gridDim.x] = sdata[4*sstride];// xxp
    }
}

//////////////////////// old kernels /////////////////////////////////////
__global__
void LossBlockSumKernel(uint* r_loss, uint* r_partial, uint r_num, uint flag)
{
  extern __shared__ uint shdmem[];
  uint indx = blockIdx.x*blockDim.x+threadIdx.x;
  if(!flag) // first time
    shdmem[threadIdx.x] = (indx<r_num) ? (r_loss[indx]!=0 ? 1 : 0) : 0;
  else    // not first time
    shdmem[threadIdx.x] = (indx<r_num) ? r_loss[indx] : 0;
  __syncthreads();
  for(int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
      shdmem[threadIdx.x] = shdmem[threadIdx.x] + shdmem[threadIdx.x+offset];
    __syncthreads();
  }
  if(threadIdx.x == 0)
    r_partial[blockIdx.x] = shdmem[0];
}

__global__
void EmittanceBlockSumKernel(double* r_x, double* r_xp, uint* r_loss, double* r_partial_x, double* r_partial_xp, double r_xoff, double r_xpoff, uint r_num, uint r_flag)
{
  extern __shared__ double shmem[];
  volatile uint indx = blockDim.x * blockIdx.x + threadIdx.x;
  if(!r_flag) // first time
  {
    shmem[threadIdx.x] = (indx<r_num) ? (r_loss[indx]==0 ? r_x[indx]-r_xoff : 0.0) : 0.0; // x
    shmem[threadIdx.x+blockDim.x] = shmem[threadIdx.x]*shmem[threadIdx.x]; // x^2
    shmem[threadIdx.x+2*blockDim.x] = (indx<r_num) ? (r_loss[indx]==0 ? r_xp[indx]-r_xpoff : 0.0) : 0.0; // xp
    shmem[threadIdx.x+3*blockDim.x] = shmem[threadIdx.x+2*blockDim.x]*shmem[threadIdx.x+2*blockDim.x]; // xp^2
    shmem[threadIdx.x+4*blockDim.x] = shmem[threadIdx.x]*shmem[threadIdx.x+2*blockDim.x]; // x*xp
  }
  else
  {
    shmem[threadIdx.x] = (indx < r_num) ? r_x[2+indx]: 0.0; // x
    shmem[threadIdx.x+blockDim.x] = (indx < r_num) ? r_x[2+indx+blockDim.x]: 0.0; // x^2
    shmem[threadIdx.x+2*blockDim.x] = (indx < r_num) ? r_xp[3+indx]: 0.0; // xp
    shmem[threadIdx.x+3*blockDim.x] = (indx < r_num) ? r_xp[3+indx+blockDim.x]: 0.0; // xp^2
    shmem[threadIdx.x+4*blockDim.x] = (indx < r_num) ? r_xp[3+indx+2*blockDim.x]: 0.0; // x*xp
  }
  __syncthreads();
  for(int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      shmem[threadIdx.x] += shmem[threadIdx.x+offset];
      shmem[threadIdx.x+blockDim.x] += shmem[threadIdx.x+blockDim.x+offset];
      shmem[threadIdx.x+2*blockDim.x] += shmem[threadIdx.x+2*blockDim.x+offset];
      shmem[threadIdx.x+3*blockDim.x] += shmem[threadIdx.x+3*blockDim.x+offset];
      shmem[threadIdx.x+4*blockDim.x] += shmem[threadIdx.x+4*blockDim.x+offset];
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!r_flag) // first time
    {
      r_partial_x[2+blockIdx.x] = shmem[0]; // x
      r_partial_x[2+gridDim.x+blockIdx.x] = shmem[blockDim.x]; //x^2
      r_partial_xp[3+blockIdx.x] = shmem[2*blockDim.x]; // xp
      r_partial_xp[3+gridDim.x+blockIdx.x] = shmem[3*blockDim.x]; // xp^2
      r_partial_xp[3+2*gridDim.x+blockIdx.x] = shmem[4*blockDim.x]; // x*xp
    }
    else
    {
      r_partial_x[blockIdx.x] = shmem[0]; // x
      r_partial_x[blockIdx.x+1] = shmem[blockDim.x]; // x^2
      r_partial_xp[blockIdx.x] = shmem[2*blockDim.x];  // xp
      r_partial_xp[blockIdx.x+1] = shmem[3*blockDim.x]; // xp^2
      r_partial_xp[blockIdx.x+2] = shmem[4*blockDim.x]; // x*xp
    }
  }
}

/*! sum of x, pad extra seats with 0.0*/
__global__
void BlockSumKernel(double* r_x, uint* r_loss, double* r_partial, uint r_num, int r_flag)
{
  extern __shared__ double bs_smem[];
  volatile uint indx = blockDim.x * blockIdx.x + threadIdx.x;
  if(!r_flag) // first time check for loss
#ifdef _BENCH
    bs_smem[threadIdx.x] = (indx<r_num) ? r_x[indx]  : 0.0;
#else
    bs_smem[threadIdx.x] = (indx<r_num) ? 
                         ((r_loss[indx]==0 ? r_x[indx] : 0.0)) : 0.0;
#endif
  else
    bs_smem[threadIdx.x] = (indx<r_num) ? r_x[indx] : 0.0;
  __syncthreads();
  for(volatile int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
      bs_smem[threadIdx.x] += bs_smem[threadIdx.x+offset];
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    r_partial[blockIdx.x] = bs_smem[0];
//    if(r_flag)
//      printf("Phase Avg Kernel: %19.17f\n", r_partial[0]);
  }
}



//   
/*! sum of x, y pad extra seats with 0.0*/
__global__
void XYBlockSumKernel(double* r_x, double* r_y, uint* r_loss, double* r_partial1,
                double* r_partial2, uint r_num, int r_flag)
{
  extern __shared__ double bs2_smem[];
  volatile uint indx = blockDim.x * blockIdx.x + threadIdx.x;
  if(!r_flag) // first time, check for loss
  {
    bs2_smem[threadIdx.x] = (indx<r_num) ?  // x
                            (r_loss[indx]==0 ? r_x[indx] : 0.0) : 0.0;
    bs2_smem[threadIdx.x+blockDim.x] = (indx<r_num) ? // y
                            (r_loss[indx]==0 ? r_y[indx] : 0.0) : 0.0;
  }
  else
  { // the first seat in partials are reserved for final avgs x, y
    bs2_smem[threadIdx.x] = (indx<r_num) ? r_x[1+indx] : 0.0;// x
    bs2_smem[threadIdx.x+blockDim.x] = (indx<r_num) ? // y
                                         r_y[1+indx] : 0.0;
  }
  __syncthreads();

  for(volatile int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      bs2_smem[threadIdx.x] += bs2_smem[threadIdx.x+offset];
      bs2_smem[threadIdx.x+blockDim.x] +=
                    bs2_smem[threadIdx.x+blockDim.x+offset];
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!r_flag) // first time
    {
      r_partial1[1+blockIdx.x] = bs2_smem[0]; // x
      r_partial2[1+blockIdx.x] = bs2_smem[blockDim.x]; // y
    }
    else
    {
      r_partial1[blockIdx.x] = bs2_smem[0]; // x
      r_partial2[blockIdx.x] = bs2_smem[blockDim.x]; // y
    }
  }
}

/*! sum of x, x^2 pad extra seats with 0.0*/
__global__
void XX2BlockSumKernel(double* r_x, uint* r_loss, double* r_partial1,
                       uint r_num, int r_flag)
{
  extern __shared__ double bsxx2_smem[];
  volatile uint indx = blockDim.x * blockIdx.x + threadIdx.x;
  if(!r_flag) // first time, check for loss
  {
    bsxx2_smem[threadIdx.x] = (indx<r_num) ?  // x
                            (r_loss[indx]==0 ? r_x[indx] : 0.0) : 0.0;
    bsxx2_smem[threadIdx.x+blockDim.x] = bsxx2_smem[threadIdx.x]  //x^2
                                      *bsxx2_smem[threadIdx.x];
  }
  else
  { // the first 2 seats in partials are reserved for final avgs x, x2
    bsxx2_smem[threadIdx.x] = (indx<r_num) ? r_x[2+indx] : 0.0;// x
    bsxx2_smem[threadIdx.x+blockDim.x] = (indx<r_num) ? // x^2
                                       r_x[2+indx+blockDim.x] : 0.0;
  }
  __syncthreads();

  for(volatile int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      bsxx2_smem[threadIdx.x] += bsxx2_smem[threadIdx.x+offset];
      bsxx2_smem[threadIdx.x+blockDim.x] +=
                    bsxx2_smem[threadIdx.x+blockDim.x+offset];
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!r_flag) // first time
    {
      r_partial1[2+blockIdx.x] = bsxx2_smem[0]; // x
      r_partial1[2+gridDim.x+blockIdx.x] = bsxx2_smem[blockDim.x]; // x^2
    }
    else
    {
      r_partial1[blockIdx.x] = bsxx2_smem[0]; // x
      r_partial1[blockIdx.x+1] = bsxx2_smem[blockDim.x]; // x^2
    }
  }
}

//   
/*! sum of x, y x^2, y^2 pad extra seats with 0.0*/
__global__
void XYX2Y2BlockSumKernel(double* r_x, double* r_y, uint* r_loss, double* r_partial1,
                double* r_partial2, uint r_num, int r_flag)
{
  extern __shared__ double bs4_smem[];
  volatile uint indx = blockDim.x * blockIdx.x + threadIdx.x;
  if(!r_flag) // first time, check for loss
  {
#ifdef _BENCH
    bs4_smem[threadIdx.x] = (indx<r_num) ?  r_x[indx] : 0.0;
#else
    bs4_smem[threadIdx.x] = (indx<r_num) ?  // x
                            (r_loss[indx]==0 ? r_x[indx] : 0.0) : 0.0;
#endif
    bs4_smem[threadIdx.x+blockDim.x] = bs4_smem[threadIdx.x]  //x^2
                                      *bs4_smem[threadIdx.x];
#ifdef _BENCH
    bs4_smem[threadIdx.x+2*blockDim.x] = (indx<r_num) ? r_y[indx] : 0.0;
#else
    bs4_smem[threadIdx.x+2*blockDim.x] = (indx<r_num) ? // y
                            (r_loss[indx]==0 ? r_y[indx] : 0.0) : 0.0;
#endif
    bs4_smem[threadIdx.x+3*blockDim.x] = bs4_smem[threadIdx.x+2*blockDim.x]// y^2
                                        *bs4_smem[threadIdx.x+2*blockDim.x];
//    if(blockIdx.x == 7)
//    printf("Block2DSumKernel input 0: x = %19.17f, x2 = %19.17f, y = %19.17f, y2 = %19.17f, index = %d\n", bs4_smem[threadIdx.x], bs4_smem[threadIdx.x+blockDim.x] , bs4_smem[threadIdx.x+2*blockDim.x],bs4_smem[threadIdx.x+3*blockDim.x], indx);
  }
  else
  { // the first 2 seats in partials are reserved for final avgs x, x2, y y2
    bs4_smem[threadIdx.x] = (indx<r_num) ? r_x[2+indx] : 0.0;// x
    bs4_smem[threadIdx.x+blockDim.x] = (indx<r_num) ? // x^2
                                       r_x[2+indx+blockDim.x] : 0.0;
    bs4_smem[threadIdx.x+2*blockDim.x] = (indx<r_num) ? // y
                                         r_y[2+indx] : 0.0;
    bs4_smem[threadIdx.x+3*blockDim.x] = (indx<r_num) ? // y^2
                                         r_y[2+indx+blockDim.x] : 0.0;
//    printf("Block2DSumKernel input 1: x = %19.17f, x2 = %19.17f, y = %19.17f, y2 = %19.17f, index = %d\n", ((indx<r_num) ? r_x[2+indx] : 0.0), ((indx<r_num)?r_x[2+indx+blockDim.x] : 0.0), ((indx<r_num)?r_y[2+indx] : 0.0),(indx<r_num)?r_y[2+indx+blockDim.x] : 0.0, indx);
  }
  __syncthreads();

  for(volatile int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      bs4_smem[threadIdx.x] += bs4_smem[threadIdx.x+offset];
      bs4_smem[threadIdx.x+blockDim.x] +=
                    bs4_smem[threadIdx.x+blockDim.x+offset];
      bs4_smem[threadIdx.x+2*blockDim.x] +=
                    bs4_smem[threadIdx.x+2*blockDim.x+offset];
      bs4_smem[threadIdx.x+3*blockDim.x] +=
                    bs4_smem[threadIdx.x+3*blockDim.x+offset];
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!r_flag) // first time
    {
      r_partial1[2+blockIdx.x] = bs4_smem[0]; // x
      r_partial1[2+gridDim.x+blockIdx.x] = bs4_smem[blockDim.x]; // x^2
      r_partial2[2+blockIdx.x] = bs4_smem[2*blockDim.x]; // y
      r_partial2[2+gridDim.x+blockIdx.x] = bs4_smem[3*blockDim.x]; // y^2
//      if(blockIdx.x == 7)
//      printf("Block2DSumKernel output 0 : x = %19.17f, x2 = %19.17f, y = %19.17f, y2 = %19.17f\n", r_partial1[2+blockIdx.x], r_partial1[2+gridDim.x+blockIdx.x], r_partial2[2+blockIdx.x], r_partial2[2+gridDim.x+blockIdx.x]);
    }
    else
    {
      r_partial1[blockIdx.x] = bs4_smem[0]; // x
      r_partial1[blockIdx.x+1] = bs4_smem[blockDim.x]; // x^2
      r_partial2[blockIdx.x] = bs4_smem[2*blockDim.x]; // y
      r_partial2[blockIdx.x+1] = bs4_smem[3*blockDim.x]; // y^2
//      printf("Block2DSumKernel output 1 : x = %19.17f, x2 = %19.17f, y = %19.17f, y2 = %19.17f\n", r_partial1[2+blockIdx.x], r_partial1[2+gridDim.x+blockIdx.x], r_partial2[2+blockIdx.x], r_partial2[2+gridDim.x+blockIdx.x]);
    }
  }
}

/*! Find the max (x-x0)^2+(y-y0)^2 and abs(z-z0), offset with xavg, yavg, phiavg
 *  Pack the extra seats with 0.0, its ok for max of abs
 */
__global__
void BlockMaxRZKernel(double* r_x, double* r_y, double* r_phi, uint* r_loss,
                double* r_xavg, double* r_yavg, double* r_phiavg,
                double* r_partialr, double* r_partialz, uint r_num, int r_flag)
{
  extern __shared__ double bmrz_smem[];
  volatile uint indx = blockDim.x * blockIdx.x + threadIdx.x;
  if(!r_flag) // first time, check for loss
  {
    if(threadIdx.x == 0)
      bmrz_smem[2*blockDim.x] = r_xavg[0];
    if(threadIdx.x == 1)
      bmrz_smem[2*blockDim.x+1] = r_yavg[0];
    if(threadIdx.x == 2)
      bmrz_smem[2*blockDim.x+2] = r_phiavg[0];
    __syncthreads();
    bmrz_smem[threadIdx.x] = (indx<r_num) ?  // r^2
         (r_loss[indx]==0 ? pow(r_x[indx]-bmrz_smem[2*blockDim.x], 2.0)+
                            pow(r_y[indx]-bmrz_smem[2*blockDim.x+1], 2.0)
          : 0.0) : 0.0;
    bmrz_smem[threadIdx.x+blockDim.x] = (indx<r_num) ? // z
        (r_loss[indx]==0 ? fabs(r_phi[indx]-bmrz_smem[2*blockDim.x+2])
        : 0.0) : 0.0;
  }
  else
  {
    bmrz_smem[threadIdx.x] = (indx<r_num) ? r_x[1+indx] : 0.0; // r^2
    bmrz_smem[threadIdx.x+blockDim.x] = (indx<r_num) ? //z 
                                     r_y[1+indx] : 0.0;
  }
  __syncthreads();
  for(volatile int offset = blockDim.x/2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {
      if(bmrz_smem[threadIdx.x] < bmrz_smem[threadIdx.x+offset]) //r^2
        bmrz_smem[threadIdx.x] = bmrz_smem[threadIdx.x+offset];
      if(bmrz_smem[threadIdx.x+blockDim.x] <
                               bmrz_smem[threadIdx.x+blockDim.x+offset]) //z
        bmrz_smem[threadIdx.x+blockDim.x] =
                                bmrz_smem[threadIdx.x+blockDim.x+offset];
    }
    __syncthreads();
  }
  if(threadIdx.x == 0)
  {
    if(!r_flag)  // first time
    {
      r_partialr[1+blockIdx.x] = bmrz_smem[0];
      r_partialz[1+blockIdx.x] = bmrz_smem[blockDim.x];
    }
    else
    {
      r_partialr[blockIdx.x] = bmrz_smem[0];
      r_partialz[blockIdx.x] = bmrz_smem[blockDim.x];
    }
  }
}

__global__
void UpdateLossKernel(uint* r_loss, uint* r_val)
{
  r_loss[0] = r_val[0];
}

__global__
void UpdateTransverseEmittanceKernel(double* r_xavg, double* r_xsig, 
  double* r_xpavg, double* r_xpsig,
  double* r_xeps, double* r_partial1, double* r_partial2, uint r_num, 
  uint* r_loss, double* r_w_ref, double r_mass)
{
  volatile double surv_num = r_num - (*r_loss);
  volatile double x = r_partial1[0]/surv_num;
  volatile double x2 = r_partial1[1]/surv_num;
  volatile double xp = r_partial2[0]/surv_num;
  volatile double xp2 = r_partial2[1]/surv_num;
  volatile double xxp = r_partial2[2]/surv_num;
  volatile double gamma1 = *r_w_ref/r_mass;
  volatile double betagamma = sqrt(gamma1*(gamma1+2.0));
  r_xavg[0] = x;
  r_xsig[0] = sqrt(x2-x*x);
  r_xpavg[0] = xp;
  r_xpsig[0] = sqrt(xp2-xp*xp);
//  r_xeps[0] = sqrt(x2*xp2-xxp*xxp)*betagamma;
  r_xeps[0] = sqrt((x2-x*x)*(xp2-xp*xp)-(xxp-x*xp)*(xxp-x*xp))*betagamma;
}

__global__
void UpdateLongitudinalEmittanceKernel(double* r_phi, double* r_phisig, 
  double* r_w, double* r_wsig, double* r_leps, double* r_partial1, 
  double* r_partial2, uint r_num, uint* r_loss)
{
  volatile double surv_num = r_num - (*r_loss);
  volatile double phi = r_partial1[0]/surv_num;
  volatile double phi2 = r_partial1[1]/surv_num;
  volatile double w = r_partial2[0]/surv_num;
  volatile double w2 = r_partial2[1]/surv_num;
  volatile double phi_w = r_partial2[2]/surv_num;
  volatile double phi_var = phi2-phi*phi;
  volatile double w_var = w2-w*w;
  volatile double phi_w_covar = phi_w-phi*w;
  r_phi[0] = phi;//-*r_phi_ref;///RADIAN;
  r_w[0] = w;//-*r_w_ref;
  r_phisig[0] = sqrt(phi_var);
  r_wsig[0] = sqrt(w_var);
  r_leps[0] = sqrt(phi_var*w_var-phi_w_covar*phi_w_covar);
//  printf("!!!! avg_phi = %f, avg_w = %f, sig_phi = %f, sig_w = %f, leps = %f\n", r_phi[0], r_w[0], r_phisig[0], r_wsig[0], r_leps[0]);
}

__global__
void UpdateMaxR2PhiKernel(double* r_r_max, double* r_abs_phi_max, 
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
  volatile double xavg = r_x_avg_val[0]/surv_num;
  volatile double yavg = r_y_avg_val[0]/surv_num;
  r_x_avg[0] = xavg;
  r_y_avg[0] = yavg;
}

__global__
void UpdateHorizontalAvgSigKernel(double* r_x_avg, double* r_x_sig,
  double* r_y_avg, double* r_y_sig, double* r_x_x2, double* r_y_y2, 
  uint r_num, uint* r_loss)
{
  volatile double surv_num = r_num - (*r_loss);
  volatile double xavg = r_x_x2[0]/surv_num;
  volatile double x2avg = r_x_x2[1]/surv_num;
  volatile double yavg = r_y_y2[0]/surv_num;
  volatile double y2avg = r_y_y2[1]/surv_num;
  r_x_avg[0] = xavg;
  r_x_sig[0] = sqrt(x2avg-xavg*xavg);
  r_y_avg[0] = yavg;
  r_y_sig[0] = sqrt(y2avg-yavg*yavg);
#ifdef _DEBUG
  printf("Horizontal scheff avg updated, x_avg=%18.16f, sum_x=%18.16f, y_avg=%18.16f, y2_avg=%18.16f, lossr=%d\n", r_x_avg[0], r_x_x2[0], r_y_avg[0], y2avg, *r_loss);
#endif
}

__global__
void UpdateVariableAvgKernel(double* r_avg, double* r_sum,
                                    uint r_num, uint* r_loss)
{
  double surv_num = r_num - (*r_loss);
  r_avg[0] = r_sum[0]/surv_num;
}

__global__
void UpdateVariableAvgWithLlossKernel(double* r_avg, double* r_sum, uint* r_num)
{
//  printf("UpdateVariableAvgWithLlossKernel, %d\n", r_num[0]);
  if(r_num[0] > 0)
    r_avg[0] = r_sum[0]/r_num[0];
  else
    r_avg[0] = -99999;
}

__global__
void UpdateVariableSigmaKernel(double* r_sig, double* r_sum,
                           double* r_sum2, uint r_num, uint* r_loss)
{
  double surv_num = r_num - (*r_loss);
  double avg = r_sum[0]/surv_num;
  double avg2 = r_sum2[0]/surv_num;
  r_sig[0] = sqrt(avg2-avg*avg);
}

__global__
void UpdateVariableSigmaWithLlossKernel(double* r_sig, double* r_sum,
                           double* r_sum2, uint* r_num)
{
  double avg = r_sum[0]/r_num[0];
  double avg2 = r_sum2[0]/r_num[0];
  r_sig[0] = sqrt(avg2-avg*avg);
}
template<typename T>
__global__
void SetValueKernel(T* r_pos, uint r_index, T r_val)
{
  r_pos[r_index] = r_val;
}

__global__
void CutBeamKernel(double* r_x, uint* r_loss, double r_min, double r_max, uint r_num)
{
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  uint stride = blockDim.x*gridDim.x;
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

template<typename T>
__global__
void ShiftVariableKernel(T* r_var, T r_val, uint r_sz)
{
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index > 0 && index < r_sz)
    r_var[index] += r_val;
}

__global__
void UpdateRelativePhiKernel(double* r_phi_r, double* r_phi, double* r_ref_phi, uint r_sz)
{
  uint index = blockIdx.x*blockDim.x + threadIdx.x;  
  if(index < r_sz)
  {
    double tmp = r_phi[index] - r_ref_phi[0] + PI;
    double sign = 1.0;
    if (tmp < 0)
    {
      sign = -1.0;
      tmp = -tmp;
    }
    r_phi_r[index] = sign*(fmod(tmp, TWOPI) - PI);
  }     
} 

__global__
void ChangeFrequencyKernel(double* r_phi, double r_freq_ratio, uint r_sz)
{
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index < r_sz)
    r_phi[index] *= r_freq_ratio;
}

__global__
void UpdateLongitudinalLossCoordinateKernel(uint* r_lloss, double* r_phi, double* r_phi_avg, uint r_sz)
{
  uint index = blockIdx.x*blockDim.x + threadIdx.x;
  if(index < r_sz)
  {
    if(r_phi[index] > r_phi_avg[0] + TWOPI + PI)
      r_lloss[index] = 1;
  }
}
#endif
