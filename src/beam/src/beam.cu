/*! 
 * \file beam.cu
 * \brief Kernel call functions for the Beam Class
 * 
 * These functions launch kernels in beam_kernel.cu and 
 * called by the functions in beam.cpp. 
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include "beam_cu.h"
#include "beam_kernel.cu"
#include "timer.h"
#include "utility.h"

/*!
 * \brief Allocate and initialize memory on device
 *
 * \callgraph
 */
void CreateBeamOnDevice(Beam* r_beam)
{
  uint num = r_beam->num_particle;
  cudaMalloc((void**)&r_beam->x, num*sizeof(double)); 
  cudaMalloc((void**)&r_beam->y, num*sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi, num*sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi_r, num*sizeof(double)); 
  cudaMalloc((void**)&r_beam->xp, num*sizeof(double)); 
  cudaMalloc((void**)&r_beam->yp, num*sizeof(double)); 
  cudaMalloc((void**)&r_beam->w, num*sizeof(double)); 
  cudaMalloc((void**)&r_beam->loss, num*sizeof(uint)); 
  cudaMalloc((void**)&r_beam->lloss, num*sizeof(uint)); 
  cudaMemset(r_beam->loss, 0, num*sizeof(uint));
  cudaMemset(r_beam->lloss, 0, num*sizeof(uint));

  cudaMalloc((void**)&r_beam->x_avg, sizeof(double)); 
  cudaMalloc((void**)&r_beam->x_avg_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->xp_avg, sizeof(double)); 
  cudaMalloc((void**)&r_beam->y_avg, sizeof(double)); 
  cudaMalloc((void**)&r_beam->y_avg_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->yp_avg, sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi_avg, sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi_avg_r, sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi_avg_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->w_avg, sizeof(double)); 
  cudaMalloc((void**)&r_beam->w_avg_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->x_sig, sizeof(double)); 
  cudaMalloc((void**)&r_beam->x_sig_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->xp_sig, sizeof(double)); 
  cudaMalloc((void**)&r_beam->y_sig, sizeof(double)); 
  cudaMalloc((void**)&r_beam->y_sig_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->yp_sig, sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi_sig, sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi_sig_r, sizeof(double)); 
  cudaMalloc((void**)&r_beam->phi_sig_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->w_sig, sizeof(double)); 
  cudaMalloc((void**)&r_beam->w_sig_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->x_emit, sizeof(double)); 
  cudaMalloc((void**)&r_beam->x_emit_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->y_emit, sizeof(double)); 
  cudaMalloc((void**)&r_beam->y_emit_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->z_emit, sizeof(double)); 
  cudaMalloc((void**)&r_beam->z_emit_good, sizeof(double)); 
  cudaMalloc((void**)&r_beam->r_max, sizeof(double)); 
  //cudaMalloc((void**)&r_beam->r_sig, sizeof(double)); 
  cudaMalloc((void**)&r_beam->abs_phi_max, sizeof(double)); 
  cudaMalloc((void**)&r_beam->num_loss, sizeof(uint));
  cudaMalloc((void**)&r_beam->num_lloss, sizeof(uint));
  cudaMalloc((void**)&r_beam->num_good, sizeof(uint));
  cudaMemset(r_beam->x_avg, 0, sizeof(double));
  cudaMemset(r_beam->x_avg_good, 0, sizeof(double));
  cudaMemset(r_beam->xp_avg, 0, sizeof(double));
  cudaMemset(r_beam->y_avg, 0, sizeof(double));
  cudaMemset(r_beam->y_avg_good, 0, sizeof(double));
  cudaMemset(r_beam->yp_avg, 0, sizeof(double));
  cudaMemset(r_beam->phi_avg, 0, sizeof(double));
  cudaMemset(r_beam->phi_avg_r, 0, sizeof(double));
  cudaMemset(r_beam->phi_avg_good, 0, sizeof(double));
  cudaMemset(r_beam->w_avg, 0, sizeof(double));
  cudaMemset(r_beam->w_avg_good, 0, sizeof(double));
  cudaMemset(r_beam->x_sig, 0, sizeof(double));
  cudaMemset(r_beam->x_sig_good, 0, sizeof(double));
  cudaMemset(r_beam->xp_sig, 0, sizeof(double));
  cudaMemset(r_beam->y_sig, 0, sizeof(double));
  cudaMemset(r_beam->y_sig_good, 0, sizeof(double));
  cudaMemset(r_beam->yp_sig, 0, sizeof(double));
  cudaMemset(r_beam->phi_sig, 0, sizeof(double));
  cudaMemset(r_beam->phi_sig_r, 0, sizeof(double));
  cudaMemset(r_beam->phi_sig_good, 0, sizeof(double));
  cudaMemset(r_beam->w_sig, 0, sizeof(double));
  cudaMemset(r_beam->w_sig_good, 0, sizeof(double));
  cudaMemset(r_beam->x_emit, 0, sizeof(double));
  cudaMemset(r_beam->x_emit_good, 0, sizeof(double));
  cudaMemset(r_beam->y_emit, 0, sizeof(double));
  cudaMemset(r_beam->y_emit_good, 0, sizeof(double));
  cudaMemset(r_beam->z_emit, 0, sizeof(double));
  cudaMemset(r_beam->z_emit_good, 0, sizeof(double));
  cudaMemset(r_beam->r_max, 0, sizeof(double));
  //cudaMemset(r_beam->r_sig, 0, sizeof(double));
  cudaMemset(r_beam->abs_phi_max, 0, sizeof(double));
  cudaMemset(r_beam->num_loss, 0, sizeof(uint));
  cudaMemset(r_beam->num_lloss, 0, sizeof(uint));
  cudaMemset(r_beam->num_good, 0, sizeof(uint));
}

/*!
 * \brief Free memory on device
 *
 * \callgraph
 */
void FreeBeamOnDevice(Beam* r_beam)
{
  cudaFree(r_beam->x);
  cudaFree(r_beam->y);
  cudaFree(r_beam->phi);
  cudaFree(r_beam->phi_r);
  cudaFree(r_beam->xp);
  cudaFree(r_beam->yp);
  cudaFree(r_beam->w);
  cudaFree(r_beam->loss);
  cudaFree(r_beam->lloss);
  cudaFree(r_beam->x_avg);
  cudaFree(r_beam->x_avg_good);
  cudaFree(r_beam->xp_avg);
  cudaFree(r_beam->y_avg);
  cudaFree(r_beam->y_avg_good);
  cudaFree(r_beam->yp_avg);
  cudaFree(r_beam->phi_avg);
  cudaFree(r_beam->phi_avg_r);
  cudaFree(r_beam->phi_avg_good);
  cudaFree(r_beam->w_avg);
  cudaFree(r_beam->w_avg_good);
  cudaFree(r_beam->x_sig);
  cudaFree(r_beam->x_sig_good);
  cudaFree(r_beam->xp_sig);
  cudaFree(r_beam->y_sig);
  cudaFree(r_beam->y_sig_good);
  cudaFree(r_beam->yp_sig);
  cudaFree(r_beam->phi_sig);
  cudaFree(r_beam->phi_sig_r);
  cudaFree(r_beam->phi_sig_good);
  cudaFree(r_beam->w_sig);
  cudaFree(r_beam->w_sig_good);
  cudaFree(r_beam->x_emit);
  cudaFree(r_beam->x_emit_good);
  cudaFree(r_beam->y_emit);
  cudaFree(r_beam->y_emit_good);
  cudaFree(r_beam->z_emit);
  cudaFree(r_beam->z_emit_good);
  cudaFree(r_beam->r_max);
  //cudaFree(r_beam->r_sig);
  cudaFree(r_beam->abs_phi_max);
  cudaFree(r_beam->num_loss);
  cudaFree(r_beam->num_lloss);
  cudaFree(r_beam->num_good);
  cudaFree(r_beam->partial_int1);
  cudaFree(r_beam->partial_double2);
  cudaFree(r_beam->partial_double3); 
}

/*!
 * \brief Allocate and initialize temporary arrays on device
 *
 * \callgraph
 */
void CreatePartialsOnDevice(Beam* r_beam, uint r_grid_size, uint r_blck_size)
{
  cudaMalloc((void**)&r_beam->partial_int1,  sizeof(uint)*(r_grid_size+1));
  cudaMalloc((void**)&r_beam->partial_double2, sizeof(double)*(r_grid_size+1)*2);
  cudaMalloc((void**)&r_beam->partial_double3, sizeof(double)*(r_grid_size+1)*3);
  cudaMemset(r_beam->partial_int1, 0, sizeof(uint)*(r_grid_size+1));
  cudaMemset(r_beam->partial_double2, 0, sizeof(double)*(r_grid_size+1)*2);
  cudaMemset(r_beam->partial_double3, 0, sizeof(double)*(r_grid_size+1)*3);
}

/*!
 * \brief Free temporary arrays on device
 *
 * \callgraph
 */
void FreePartialsOnDevice(Beam* r_beam)
{
  cudaFree(r_beam->partial_int1);
  cudaFree(r_beam->partial_double2);
  cudaFree(r_beam->partial_double3); 
}

/*!
 * \brief Update beam coordinates on device based on a distribution
 *  on host.
 *
 * \callgraph
 */
void UpdateBeamOnDevice(Beam* r_beam, double* r_x_h, double* r_xp_h, 
  double* r_y_h, double* r_yp_h, double* r_phi_h, double* r_w_h, 
  uint* r_loss_h, uint* r_lloss_h)
{
  uint num = r_beam->num_particle;
  cudaMemcpy(r_beam->x, r_x_h, sizeof(double)*num, cudaMemcpyHostToDevice);
  cudaMemcpy(r_beam->xp, r_xp_h, sizeof(double)*num, cudaMemcpyHostToDevice);
  cudaMemcpy(r_beam->y, r_y_h, sizeof(double)*num, cudaMemcpyHostToDevice);
  cudaMemcpy(r_beam->yp, r_yp_h, sizeof(double)*num, cudaMemcpyHostToDevice);
  cudaMemcpy(r_beam->phi, r_phi_h, sizeof(double)*num, cudaMemcpyHostToDevice);
  cudaMemcpy(r_beam->w, r_w_h, sizeof(double)*num, cudaMemcpyHostToDevice); 
  if(r_loss_h == NULL)
    cudaMemset(r_beam->loss, 0, num*sizeof(uint));
  else
    cudaMemcpy(r_beam->loss, r_loss_h, sizeof(uint)*num, 
      cudaMemcpyHostToDevice);
  if(r_lloss_h == NULL)
    cudaMemset(r_beam->lloss, 0, num*sizeof(uint));
  else
    cudaMemcpy(r_beam->lloss, r_lloss_h, sizeof(uint)*num, 
      cudaMemcpyHostToDevice); 
}

/*!
 * \brief Copy beam coordinates from device to host.
 *
 * Used for output.
 *
 * \callgraph
 */
void CopyBeamFromDevice(Beam* r_beam, double* r_x_h, double* r_xp_h, 
  double* r_y_h, double* r_yp_h, double* r_phi_h, double* r_w_h, 
  uint* r_loss_h, uint* r_lloss_h, uint* r_num_loss_h)
{
  uint num = r_beam->num_particle;
  cudaMemcpy(r_x_h, r_beam->x, sizeof(double)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_xp_h, r_beam->xp, sizeof(double)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_y_h, r_beam->y, sizeof(double)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_yp_h, r_beam->yp, sizeof(double)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_phi_h, r_beam->phi, sizeof(double)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_w_h, r_beam->w, sizeof(double)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_loss_h, r_beam->loss, sizeof(uint)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_lloss_h, r_beam->lloss, sizeof(uint)*num, cudaMemcpyDeviceToHost);
  cudaMemcpy(r_num_loss_h, r_beam->num_loss, sizeof(uint), 
    cudaMemcpyDeviceToHost);
}

/*!
 * \brief Copy a particle's coordinates from device to host.
 *
 * Used for output.
 *
 * \callgraph
 */
void CopyParticleFromDevice(Beam* r_beam, uint r_index, double* r_x_h, 
  double* r_xp_h, double* r_y_h, double* r_yp_h, double* r_phi_h, double* r_w_h,
  uint* r_loss_h, uint* r_lloss_h)
{
  cudaMemcpy(r_x_h, r_beam->x+r_index, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r_xp_h, r_beam->xp+r_index, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r_y_h, r_beam->y+r_index, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r_yp_h, r_beam->yp+r_index, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r_phi_h, r_beam->phi+r_index, sizeof(double), 
    cudaMemcpyDeviceToHost);
  cudaMemcpy(r_w_h, r_beam->w+r_index, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r_loss_h, r_beam->loss+r_index, sizeof(uint), 
    cudaMemcpyDeviceToHost);
  cudaMemcpy(r_lloss_h, r_beam->lloss+r_index, sizeof(uint), 
    cudaMemcpyDeviceToHost);
}

/*!
 * \brief Get an optimized block and grid setting.
 *
 * Used by functions within this file.
 *
 * \callgraph
 */
void GetNumBlocksAndThreads(uint r_sz, uint r_max_blocks, uint r_max_threads, 
                            uint& r_grid_size, uint& r_block_size)
{
  r_block_size = (r_sz < r_max_threads*2) ? NextPow2((r_sz + 1) / 2) : 
    r_max_threads;
  r_grid_size = (r_sz + (r_block_size * 2 - 1)) / (r_block_size * 2);
  r_grid_size = std::min(r_max_blocks, r_grid_size);
}

/*!
 * \brief Call the parallel reduction kernel to calculate the 
 *        sum of a beam coordinate.
 *
 * \callgraph
 * \callergraph
 */
template<class T>
void XReduceKernelCall(uint r_block_size, uint r_grid_size, T* r_idata, 
  T* r_odata, uint r_size, Beam* r_beam, uint r_first_pass_flag, 
  bool check_lloss = false)
{
  uint smem_sz = (r_block_size <= 32) ? 64 * sizeof(T) : r_block_size*sizeof(T);
  if(check_lloss)
  {
    switch (r_block_size)
    {
      case 512:
          XReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 256:
          XReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 128:
          XReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 64:
          XReduceKernel<T,  64><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 32:
          XReduceKernel<T,  32><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 16:
          XReduceKernel<T,  16><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  8:
          XReduceKernel<T,   8><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  4:
          XReduceKernel<T,   4><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  2:
          XReduceKernel<T,   2><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  1:
          XReduceKernel<T,   1><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break; 
    }  
  }  
  else
  {
    switch (r_block_size)
    {
      case 512:
          XReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 256:
          XReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 128:
          XReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 64:
          XReduceKernel<T,  64><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 32:
          XReduceKernel<T,  32><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 16:
          XReduceKernel<T,  16><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  8:
          XReduceKernel<T,   8><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  4:
          XReduceKernel<T,   4><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  2:
          XReduceKernel<T,   2><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  1:
          XReduceKernel<T,   1><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break; 
    }
  }
}

/*!
 * \brief Update the average of a beam coordindate.
 *
 * \param r_beam Pointer to the beam
 * \param r_x Target beam coordinate
 * \pram r_x_avg[out] Target beam coordinate average
 * \param r_good_only If false(default), consider particles which are not 
 *                    transversely lost, otherwise, only consider good particles
 *                    which are not lost either transversely nor longitudinally.
 */
void UpdateAvgOfOneVariableKernelCall(Beam* r_beam, double* r_x, 
  double* r_x_avg, bool r_good_only)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  XReduceKernelCall<double>(block_sz, grid_sz, r_x, partial1, num, r_beam, 1, 
    r_good_only);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    XReduceKernelCall<double>(block_sz, grid_sz, partial1, partial1, s, r_beam, 
      0, r_good_only); 
    s = (s + (block_sz*2-1))/(block_sz*2);
  }
  //StopTimer(&start, &stop, "new reducion kernel-2");
  if(!r_good_only)
    UpdateVariableAvgKernel<<<1, 1>>>(r_x_avg, partial1, num, r_beam->num_loss);
  else
    UpdateVariableAvgWithLlossKernel<<<1, 1>>>(r_x_avg, partial1, 
      r_beam->num_good);
}

/*!
 * \brief Call parallel reduce kernel to count the number of non-zero elements 
 * 	of an uint array.
 *
 * \callgraph
 * \callergraph
 */
void LossReduceKernelCall(uint r_block_size, uint r_grid_size, uint* r_idata, 
                          uint* r_odata, uint r_size, uint r_first_pass_flag)
{
  uint smem_sz = (r_block_size <= 32) ? 64 * sizeof(uint) : 
                                        r_block_size*sizeof(uint); 
  switch (r_block_size)
  {
    case 512:
        LossReduceKernel<512><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case 256:
        LossReduceKernel<256><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case 128:
        LossReduceKernel<128><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case 64:
        LossReduceKernel<64><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case 32:
        LossReduceKernel<32><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case 16:
        LossReduceKernel<16><<< r_grid_size, r_block_size, smem_sz >>>(  
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case  8:
        LossReduceKernel<8><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case  4:
        LossReduceKernel<4><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case  2:
        LossReduceKernel<2><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
    case  1:
        LossReduceKernel<1><<< r_grid_size, r_block_size, smem_sz >>>(
            r_idata, r_odata, r_size, r_first_pass_flag); break;
  }  
}

/*!
 * \brief Update the number of particles lost
 *
 * \param r_beam Pointer to a beam object
 * \param r_lloss Flag to indicate transverse(false) or longitudinal(true). 
 * \callgraph
 * \callergraph
 */
void UpdateLossCountKernelCall(Beam* r_beam, bool r_lloss)
{
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  uint* partial = r_beam->partial_int1;
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  if(!r_lloss)
    LossReduceKernelCall(block_sz, grid_sz, r_beam->loss, partial, num, 1);
  else
    LossReduceKernelCall(block_sz, grid_sz, r_beam->lloss, partial, num, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    LossReduceKernelCall(block_sz, grid_sz, partial, partial, s, 0);
    s = (s + (block_sz * 2 - 1)) / (block_sz * 2);
  }
  if(!r_lloss)
    cudaMemcpy(r_beam->num_loss, partial, sizeof(uint), 
      cudaMemcpyDeviceToDevice);
  else
    cudaMemcpy(r_beam->num_lloss, partial, sizeof(uint), 
      cudaMemcpyDeviceToDevice);
}

/*!
 * \brief Call parallel reduce kernel to calculate the sum of x and x^2 of 
 *	coordinates. 
 *
 * \callgraph
 * \callergraph
 */
template<class T>
void XX2ReduceKernelCall(uint r_block_size, uint r_grid_size, T* r_idata, 
  T* r_odata, uint r_size, Beam* r_beam, uint r_first_pass_flag, 
  bool check_lloss = false)
{
  uint smem_sz = (r_block_size <= 32) ? 64 * sizeof(T) : r_block_size * 
		  sizeof(T);
  smem_sz *= 2;

  if(check_lloss)
  {
    switch (r_block_size)
    {
      case 512:
          XX2ReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 256:
          XX2ReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 128:
          XX2ReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 64:
          XX2ReduceKernel<T, 64><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 32:
          XX2ReduceKernel<T, 32><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case 16:
          XX2ReduceKernel<T, 16><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  8:
          XX2ReduceKernel<T, 8><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  4:
          XX2ReduceKernel<T, 4><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  2:
          XX2ReduceKernel<T, 2><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
      case  1:
          XX2ReduceKernel<T, 1><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, 
	    r_beam->lloss); 
	  break;
    }  
  }
  else
  {
    switch (r_block_size)
    {
      case 512:
          XX2ReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 256:
          XX2ReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 128:
          XX2ReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 64:
          XX2ReduceKernel<T, 64><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 32:
          XX2ReduceKernel<T, 32><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case 16:
          XX2ReduceKernel<T, 16><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  8:
          XX2ReduceKernel<T, 8><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  4:
          XX2ReduceKernel<T, 4><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  2:
          XX2ReduceKernel<T, 2><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
      case  1:
          XX2ReduceKernel<T, 1><<< r_grid_size, r_block_size, smem_sz >>>(
	    r_idata, r_odata, r_size, r_beam->loss, r_first_pass_flag, NULL); 
	  break;
    }
  }
}

/*!
 * \brief Update the std of a beam coordindate.
 *
 * \param r_beam Pointer to the beam
 * \param r_x Target beam coordinate
 * \pram r_x_sig[out] Target beam coordinate average
 * \param r_good_only If false(default), consider particles which are not 
 *                    transversely lost, otherwise, only consider good particles
 *                    which are not lost either transversely nor longitudinally.
 *
 * \callgraph
 * \callergraph
 */
void UpdateSigmaOfOneVariableKernelCall(Beam* r_beam, double* r_x, 
    double* r_x_sig, bool r_good_only)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  XX2ReduceKernelCall<double>(block_sz, grid_sz, r_x, partial1, num, r_beam, 1, 
    r_good_only);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    XX2ReduceKernelCall<double>(block_sz, grid_sz, partial1, partial1, s, 
      r_beam, 0, r_good_only); 
    s = (s + (block_sz*2-1))/(block_sz*2);
  }

  if(!r_good_only)
    UpdateVariableSigmaKernel<<<1, 1>>>(r_x_sig, partial1, 
                      partial1+1, num, r_beam->num_loss);
  else
    UpdateVariableSigmaWithLlossKernel<<<1, 1>>>(r_x_sig, partial1, partial1+1, 
                      r_beam->num_good);
  //StopTimer(&start, &stop, "new sig reducion kernel");
}


template<class T>
void XYReduceKernelCall(uint r_block_size, uint r_grid_size, T* r_idata1, 
      T* r_idata2, T* r_odata1, T* r_odata2, uint r_size, uint* r_loss, 
      uint r_first_pass_flag)
{
  uint smem_sz = (r_block_size <= 32) ? 64*sizeof(T) : r_block_size*sizeof(T); 
  smem_sz *= 2;
  switch (r_block_size)
  {
    case 512:
        XYReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 256:
        XYReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 128:
        XYReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 64:
        XYReduceKernel<T, 64><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 32:
        XYReduceKernel<T, 32><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 16:
        XYReduceKernel<T, 16><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  8:
        XYReduceKernel<T, 8><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  4:
        XYReduceKernel<T, 4><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  2:
        XYReduceKernel<T, 2><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  1:
        XYReduceKernel<T, 1><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
  }  
}

/*!
 * \brief Calculate averages for x and y coordinates simultaneously
 *
 * \callgraph
 * \callergraph
 */
void UpdateAvgXYKernelCall(Beam* r_beam)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  double* partial2 = r_beam->partial_double3;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  XYReduceKernelCall<double>(block_sz, grid_sz, r_beam->x, r_beam->y, partial1, 
                              partial2, num, r_beam->loss, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    XYReduceKernelCall<double>(block_sz, grid_sz, partial1, partial2, 
                                partial1, partial2, s, NULL, 0); 
    s = (s + (block_sz*2-1))/(block_sz*2);
  }
  UpdateHorizontalAvgXYKernel<<<1, 1>>>(r_beam->x_avg, 
    r_beam->y_avg, partial1, partial2, num, r_beam->num_loss);
  //StopTimer(&start, &stop, "new XY reducion kernel");
}

/*!
 * \brief Call kernel functions to update max r^2 = x^2+y^2 and the absolute
 *        phase
 * 
 * \callgraph 
 * \callergraph
 */
template<class T>
void MaxR2PhiReduceKernelCall(uint r_block_size, uint r_grid_size, T* r_idata1, 
      T* r_idata2, T* r_idata3, uint* r_loss, T* r_x_avg, T* r_y_avg, 
      T* r_phi_avg, T* r_odata1, T* r_odata2, uint r_size, 
      uint r_first_pass_flag)
{
  uint smem_sz = (r_block_size <= 32) ? 64*sizeof(T) : r_block_size*sizeof(T); 
  smem_sz *= 2;
  switch (r_block_size)
  {
    case 512:
        MaxR2PhiReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case 256:
        MaxR2PhiReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case 128:
        MaxR2PhiReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case 64:
        MaxR2PhiReduceKernel<T, 64><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case 32:
        MaxR2PhiReduceKernel<T, 32><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case 16:
        MaxR2PhiReduceKernel<T, 16><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case  8:
        MaxR2PhiReduceKernel<T, 8><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case  4:
        MaxR2PhiReduceKernel<T, 4><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case  2:
        MaxR2PhiReduceKernel<T, 2><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
    case  1:
        MaxR2PhiReduceKernel<T, 1><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_idata3, r_loss, r_x_avg, r_y_avg, r_phi_avg, 
          r_odata1, r_odata2, r_size,  r_first_pass_flag); break;
  }  
}

/*!
 * \brief Update the maximum r = sqrt(x^2+y^2) and absolute phase
 * 
 * \callgraph
 * \callergraph
 */
void UpdateMaxRPhiKernelCall(Beam* r_beam)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  double* partial2 = r_beam->partial_double3;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  MaxR2PhiReduceKernelCall<double>(block_sz, grid_sz, r_beam->x, r_beam->y, 
    r_beam->phi_r, r_beam->loss, r_beam->x_avg, r_beam->y_avg, r_beam->phi_avg_r,
    partial1, partial2, num, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    MaxR2PhiReduceKernelCall<double>(block_sz, grid_sz, partial1, partial2, 
      NULL, NULL, NULL, NULL, NULL, partial1, partial2, s, 0);
    s = (s + (block_sz*2-1))/(block_sz*2);
  }
  UpdateMaxRPhiKernel<<<1, 1>>>(r_beam->r_max, r_beam->abs_phi_max, 
                                  partial1, partial2);
  //StopTimer(&start, &stop, "new MaxR2Phi reducion kernel");
}

/*!
 * \brief Call kernel function to calculates averages and stds of x and y 
 *	coordiantes simultaneously. 
 *
 * \callgraph
 * \callergraph
 */
template<class T>
void XYX2Y2ReduceKernelCall(uint r_block_size, uint r_grid_size, T* r_idata1, 
      T* r_idata2, T* r_odata1, T* r_odata2, uint r_size, uint* r_loss, 
      uint* r_lloss, uint r_first_pass_flag)
{
  uint smem_sz = (r_block_size <= 32) ? 64 * sizeof(T) : r_block_size*sizeof(T);
  smem_sz *= 4;
  switch (r_block_size)
  {
    case 512:
        XYX2Y2ReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss,
          r_first_pass_flag); break;
    case 256:
        XYX2Y2ReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case 128:
        XYX2Y2ReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case 64:
        XYX2Y2ReduceKernel<T, 64><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case 32:
        XYX2Y2ReduceKernel<T, 32><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case 16:
        XYX2Y2ReduceKernel<T, 16><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case  8:
        XYX2Y2ReduceKernel<T, 8><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case  4:
        XYX2Y2ReduceKernel<T, 4><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case  2:
        XYX2Y2ReduceKernel<T, 2><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
    case  1:
        XYX2Y2ReduceKernel<T, 1><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, r_lloss, 
          r_first_pass_flag); break;
  }  
}

/*!
 * \brief Calculates averages and stds of x and y coordiantes simultaneously. 
 * 
 * \callgraph
 * \callergraph
 */
void UpdateAvgSigXYKernelCall(Beam* r_beam)
{
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  double* partial2 = r_beam->partial_double3;
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  XYX2Y2ReduceKernelCall<double>(block_sz, grid_sz, r_beam->x, r_beam->y, 
    partial1, partial2, num, r_beam->loss, r_beam->lloss, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    XYX2Y2ReduceKernelCall<double>(block_sz, grid_sz, partial1, partial2, 
                                partial1, partial2, s, NULL, NULL, 0); 
    s = (s + (block_sz * 2 - 1)) / (block_sz * 2);
  } 
  UpdateHorizontalAvgSigKernel<<<1, 1>>>(r_beam->x_avg_good, r_beam->x_sig_good,
    r_beam->y_avg_good, r_beam->y_sig_good, partial1, partial2, num, 
    r_beam->num_loss);
}

/*!
 * \brief Call parallel reduce kernel to calculate the sum of 
 *        x, x^2, xp, xp^2 and xxp saverage imultaneously
 * 
 * \callgraph
 * \callergraph
 */
template<class T>
void EmittanceReduceKernelCall(uint r_block_size, uint r_grid_size, T* r_idata1,
      T* r_idata2, T* r_odata1, T* r_odata2, uint r_size, uint* r_loss, 
      uint r_first_pass_flag)
{
  uint smem_sz = (r_block_size <= 32) ? 64*sizeof(T) : r_block_size*sizeof(T); 
  smem_sz *= 5;
  switch (r_block_size)
  {
    case 512:
        EmittanceReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 256:
        EmittanceReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 128:
        EmittanceReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 64:
        EmittanceReduceKernel<T, 64><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 32:
        EmittanceReduceKernel<T, 32><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case 16:
        EmittanceReduceKernel<T, 16><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  8:
        EmittanceReduceKernel<T, 8><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  4:
        EmittanceReduceKernel<T, 4><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  2:
        EmittanceReduceKernel<T, 2><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
    case  1:
        EmittanceReduceKernel<T, 1><<< r_grid_size, r_block_size, smem_sz >>>(
          r_idata1, r_idata2, r_odata1, r_odata2, r_size, r_loss, 
          r_first_pass_flag); break;
  }  
}


/*!
 * \brief Update the horizontal emittance of a beam
 * 
 * \callgraph
 * \callergraph
 */
void UpdateHorizontalEmittanceKernelCall(Beam* r_beam)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  double* partial2 = r_beam->partial_double3;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  EmittanceReduceKernelCall<double>(block_sz, grid_sz, r_beam->x, r_beam->xp, 
                                  partial1, partial2, num, r_beam->loss, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    EmittanceReduceKernelCall<double>(block_sz, grid_sz, partial1, partial2, 
                                      partial1, partial2, s, NULL, 0); 
    s = (s + (block_sz * 2 - 1)) / (block_sz * 2);
  }
  UpdateTransverseEmittanceKernel<<<1, 1>>>(r_beam->x_avg, r_beam->x_sig, 
    r_beam->xp_avg, r_beam->xp_sig, r_beam->x_emit, 
    partial1, partial2, num, r_beam->num_loss, r_beam->w, r_beam->mass);
  //StopTimer(&start, &stop, "new h-emittance reducion kernel");
}

/*!
 * \brief Update the vertical emittance of a beam
 * 
 * \callgraph
 * \callergraph
 */
void UpdateVerticalEmittanceKernelCall(Beam* r_beam)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  double* partial2 = r_beam->partial_double3;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  EmittanceReduceKernelCall<double>(block_sz, grid_sz, r_beam->y, r_beam->yp, 
                                  partial1, partial2, num, r_beam->loss, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    EmittanceReduceKernelCall<double>(block_sz, grid_sz, partial1, partial2, 
                                      partial1, partial2, s, NULL, 0); 
    s = (s + (block_sz * 2 - 1)) / (block_sz * 2);
  }
  UpdateTransverseEmittanceKernel<<<1, 1>>>(r_beam->y_avg, r_beam->y_sig, 
    r_beam->yp_avg, r_beam->yp_sig, r_beam->y_emit, 
    partial1, partial2, num, r_beam->num_loss, r_beam->w, r_beam->mass);
  //StopTimer(&start, &stop, "new v-emittance reducion kernel");
}

/*!
 * \brief Update the longitudinal emittance of a beam
 * 
 * \callgraph
 * \callergraph
 */
void UpdateLongitudinalEmittanceKernelCall(Beam* r_beam)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  double* partial2 = r_beam->partial_double3;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  EmittanceReduceKernelCall<double>(block_sz, grid_sz, r_beam->phi_r, r_beam->w,
                                  partial1, partial2, num, r_beam->loss, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    EmittanceReduceKernelCall<double>(block_sz, grid_sz, partial1, partial2, 
                                      partial1, partial2, s, NULL, 0); 
    s = (s + (block_sz * 2 - 1)) / (block_sz * 2);
  }
  UpdateLongitudinalEmittanceKernel<<<1, 1>>>(r_beam->phi_avg_r, 
    r_beam->phi_sig_r, r_beam->w_avg, r_beam->w_sig, r_beam->z_emit, partial1, 
    partial2, num, r_beam->num_loss);
  //StopTimer(&start, &stop, "new l-emittance reducion kernel");
}

/*!
 * \brief Set value to an element in a double array
 * 
 * \param r_arr Array pointer
 * \param r_index Array index of the element
 * \param r_val Element value.
 */
void SetDoubleValue(double* r_arr, uint r_index, double r_val)
{
  SetValueKernel<<<1, 1>>>(r_arr, r_index, r_val);
}

void ShiftVariableKernelCall(Beam* r_beam, double* r_var, double r_val)
{
  ShiftVariableKernel<double><<<r_beam->grid_size, r_beam->blck_size>>>(r_var, 
    r_val, r_beam->num_particle);
}

/*!
 * \brief Update the relative phase coordinates
 * 
 * \param r_beam A pointer to beam
 * \param r_use_good If true, the reference phase used is the average 
 *        of the good particles, otherwise, relative to the reference 
 *        particle's phase
 * \callgraph
 * \callergraph
 */
void UpdateRelativePhiKernelCall(Beam* r_beam, bool r_use_good)
{
  if(!r_use_good)
    UpdateRelativePhiKernel<<<r_beam->grid_size, r_beam->blck_size>>>(
      r_beam->phi_r, r_beam->phi, r_beam->phi, r_beam->num_particle);
  else
    UpdateRelativePhiKernel<<<r_beam->grid_size, r_beam->blck_size>>>(
      r_beam->phi_r, r_beam->phi, r_beam->phi_avg_good, r_beam->num_particle);
}

/*!
 * \brief Cut beam
 * 
 * \callgraph
 * \callergraph
 */
void CutBeamKernelCall(double* r_coord, uint* r_loss, double r_min, double r_max,
  uint r_num, uint r_grid, uint r_blck)
{
  CutBeamKernel<<<r_grid, r_blck>>>(r_coord, r_loss, r_min, r_max, r_num);
}

/*!
 * \brief Update beam absolute phase once the frequency has changed
 *
 * \callgraph
 * \callergraph
 */
void ChangeFrequnecyKernelCall(Beam* r_beam, double r_freq_ratio)
{
  ChangeFrequencyKernel<<<r_beam->grid_size, r_beam->blck_size>>>(r_beam->phi, 
    r_freq_ratio, r_beam->num_particle);
}

/*!
 * \brief Call kernel function to update longitudinal coordinates
 *
 * \callgraph
 * \callergraph
 */
void UpdateLongitudinalLossCoordinateKernelCall(Beam* r_beam)
{
  UpdateLongitudinalLossCoordinateKernel<<<r_beam->grid_size, 
    r_beam->blck_size>>>(r_beam->lloss, r_beam->phi, r_beam->phi_avg_good, 
    r_beam->num_particle);
}

/*!
 * \brief Call Parallel reduce kernel to count the number of good particles 
 *	(not lost neither transversely nor longitudinally)
 *
 * \callgraph
 * \callergraph
 */
void GoodParticleCountReduceKernelCall(uint r_block_size, uint r_grid_size, 
    uint* r_idata, uint* r_idata2, uint* r_odata, uint r_size, 
    uint r_first_pass_flag)
{
  uint smem_sz = (r_block_size <= 32) ? 64 * sizeof(uint) : 
                                        r_block_size * sizeof(uint); 
  switch (r_block_size)
  {
    case 512:
        GoodParticleCountReduceKernel<512><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case 256:
        GoodParticleCountReduceKernel<256><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case 128:
        GoodParticleCountReduceKernel<128><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case 64:
        GoodParticleCountReduceKernel<64><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case 32:
        GoodParticleCountReduceKernel<32><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case 16:
        GoodParticleCountReduceKernel<16><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case  8:
        GoodParticleCountReduceKernel<8><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case  4:
        GoodParticleCountReduceKernel<4><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case  2:
        GoodParticleCountReduceKernel<2><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
    case  1:
        GoodParticleCountReduceKernel<1><<< r_grid_size, r_block_size, 
	  smem_sz >>>(r_idata, r_idata2, r_odata, r_size, r_first_pass_flag); 
	break;
  }  
}

/*!
 * \brief Count the number of good particles
 *
 * \callgraph
 * \callergraph
 */
void UpdateGoodParticleCountKernelCall(Beam* r_beam)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  uint* partial = r_beam->partial_int1;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  GoodParticleCountReduceKernelCall(block_sz, grid_sz, r_beam->loss, 
    r_beam->lloss, partial, num, 1);
  //StopTimer(&start, &stop, "new loss reducion kernel-1");
  //StartTimer(&start, &stop);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    GoodParticleCountReduceKernelCall(block_sz, grid_sz, partial, NULL, 
	partial, s, 0);
    s = (s + (block_sz * 2 - 1)) / (block_sz * 2);
  }
  cudaMemcpy(r_beam->num_good, partial, sizeof(uint), cudaMemcpyDeviceToDevice);
  //StopTimer(&start, &stop, "new loss reducion kernel");
}

/*
template<class T>
void RR2ReduceKernelCall(uint r_block_size, uint r_grid_size, T* r_idatax, 
    T* r_idatay, T* r_odata, uint r_size, Beam* r_beam, uint r_first_pass_flag)
{
  uint smem_sz = (r_block_size <= 32) ? 64*sizeof(T) : r_block_size*sizeof(T); 
  smem_sz *= 2;
  switch (r_block_size)
  {
    case 512:
        RR2ReduceKernel<T, 512><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case 256:
        RR2ReduceKernel<T, 256><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case 128:
        RR2ReduceKernel<T, 128><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case 64:
        RR2ReduceKernel<T, 64><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case 32:
        RR2ReduceKernel<T, 32><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case 16:
        RR2ReduceKernel<T, 16><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case  8:
        RR2ReduceKernel<T, 8><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case  4:
        RR2ReduceKernel<T, 4><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case  2:
        RR2ReduceKernel<T, 2><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
    case  1:
        RR2ReduceKernel<T, 1><<< r_grid_size, r_block_size, smem_sz >>>(
	  r_idatax, r_idatay, r_odata, r_size, r_beam->loss, r_first_pass_flag);
	break;
  }  
}

void UpdateSigmaR(Beam* r_beam)
{
  //cudaEvent_t start, stop; 
  uint block_sz, grid_sz;
  uint num = r_beam->num_particle;
  double* partial1 = r_beam->partial_double2;
  //StartTimer(&start, &stop);
  GetNumBlocksAndThreads(num, 64, r_beam->blck_size, grid_sz, block_sz);
  RR2ReduceKernelCall<double>(block_sz, grid_sz, r_beam->x, r_beam->y, 
      partial1, num, r_beam, 1);
  uint s = grid_sz;
  while(s > 1)
  {
    GetNumBlocksAndThreads(s, 64, r_beam->blck_size, grid_sz, block_sz);
    RR2ReduceKernelCall<double>(block_sz, grid_sz, partial1, NULL, partial1, 	
      s, r_beam, 0); 
    s = (s + (block_sz * 2 - 1)) / (block_sz * 2);
  }
  UpdateVariableSigmaKernel<<<1, 1>>>(r_beam->r_sig, partial1, 
                      partial1+1, num, r_beam->num_loss);
  //StopTimer(&start, &stop, "new sig reducion kernel");
}
*/
