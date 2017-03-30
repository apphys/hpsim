#ifndef SPACE_CHARGE_KERNEL_CU
#define SPACE_CHARGE_KERNEL_CU

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include "space_charge_parameter.h"
#include "beamline_element.h"
#include "beam.h"
#include "constant.h"

__device__ ScheffMeshSize scheff_sz;
__device__ uint num_on_mesh;
//__constant__ SpaceChargeParam d_scheff_param;
__device__ SpaceChargeParam d_scheff_param;

/*! TODO If dz > beta_lambda, set it to 0.99bl ?*/
__global__
void UpdateScheffMeshSizeKernel(double* r_partial_r, double* r_partial_z,
  double r_freq, double r_beta, double r_coef = 1.0)
{
  double bl = r_beta * CLIGHT / r_freq;
  scheff_sz.dr = r_coef * r_partial_r[0] / d_scheff_param.nx;
  if(r_coef == 1.0)
    scheff_sz.dz = bl / d_scheff_param.nz;
  else
  {
    if (r_partial_z[0] * r_coef > PI) 
      scheff_sz.dz = bl / d_scheff_param.nz;
    scheff_sz.dz = r_coef * bl * r_partial_z[0] / (d_scheff_param.nz * PI);
  }
}

// blocksz(64, 8, 1) total thread num < 1024
// gridsz (4, 33, 1) 
// blockDim.x = 64(zp), blockDim.y = 8(rs), gridDim.x = 4(rs) gridDim.y = 33(rp) 
// threadIdx.x(0-63), threadIdx.y(0-7),blockIdx.x(0-3), blockIdx.y(0-32), 
__global__
void UpdateFldTbl1Kernel(double2* r_fld_tbl1, double r_freq, double r_beta, 
                uint r_num_part, double r_current_ov_mc2, uint r_adj_bunch)
{
  double gm = rsqrt(1.0 - r_beta * r_beta);
  double dr = scheff_sz.dr;
  double dzg = scheff_sz.dz * gm; // use dz as dzg for now.
  uint k = blockDim.y * blockIdx.x + threadIdx.y; // 0-31, source r coordinate
  double rs = sqrt(0.5 * (k * k + (k + 1) * (k + 1))) * dr; // source r
  double zs = 0.5 * dzg; // source z
  double zp = (threadIdx.x + 1) * dzg;
  double rp = blockIdx.y * dr; // 0-32, receiving point r
  uint blocksz = blockDim.x * blockDim.y;
  uint cnt = blockIdx.y * gridDim.x * blocksz + blockIdx.x * blocksz
                      + threadIdx.y * blockDim.x + threadIdx.x;
  double d = zp - zs;
  double d2 = d * d;
  double c = (rp - rs) * (rp - rs);
  double b = (rp + rs) * (rp + rs);
  double rsrp = rs * rp;
  double a = 4.0 * rsrp / (b + d2);
  double g = 1.0 - a;
  double h = log(g);
  double ee=1.0 + g * (0.4630106 - 0.2452740 * h + g * 
	   (0.1077857 - 0.04125321 * h));
  double ek=1.38629436 - 0.5 * h + g * (0.1119697 - 0.1213486 * h + g * 
	  (0.07253230 - 0.02887472 * h));
  double er1 = 0.0;
  a = sqrt(b + d2);
  double rs2mrp2 = (rs * rs - rp * rp);
  if(rp != 0.0)
    er1 = (ek - (rs2mrp2 + d2) * ee / (c + d2)) / (2.0 * rp * a);
  double ez1 = d * ee / (a * (c + d2));
  double wave_len = CLIGHT / r_freq;
  double pl = r_beta * gm * wave_len;
  double xi, d1;
  for(int i = 1; i <= r_adj_bunch; ++i)
  {
    xi = i;
    for(int j = 0; j < 2; ++j)
    {
      d1 = d - xi * pl;
      d2 = d1 * d1;
      a = 4.0 * rsrp / (b + d2);
      g = 1.0 - a;
      h = log(g);
      ee = 1.0 + g * (0.4630106 - 0.2452740 * h + g * 
	  (0.1077857 - 0.04125321 * h));
      ek = 1.38629436 - 0.5 * h + g * (0.1119697 - 0.1213486 * h + g * 
	  (0.07253230 - 0.02887472 * h));
      a = sqrt(b + d2);
      if(rp != 0.0) 
        er1 += (ek - (rs2mrp2 + d2) * ee / (c + d2)) / (2.0 * rp * a);
      ez1 += d1 * ee / (a * (c + d2));
      xi = -xi;
    }
  }
  //TODO: Where's 5721.67 from? Change this!!! 
  // 1/(2*pi^2*epsilon)= 5721.67m*mev/coul
  double c1 = 5721.67 * r_current_ov_mc2 / (r_freq * 1e6 * (double)r_num_part);
  r_fld_tbl1[cnt].x = c1 * er1;
  r_fld_tbl1[cnt].y = c1 * ez1;
}

__global__
void InitNumOnMesh()
{
  num_on_mesh = 0;
}

__device__ 
double atomicAddDouble(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;  
  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, 
            __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__
void DistributeParticleKernel(
#ifdef DOUBLE_PRECISION
  double* r_bin_tbl,
#else
  float* r_bin_tbl, 
#endif
  double* r_x, double* r_y, 
  double* r_phi, uint* r_loss, uint* r_lloss, double* r_x_avg, 
  double* r_y_avg, double* r_phi_avg, double* r_x_sig, double* r_y_sig, 
  uint r_num_part, double r_beta, double r_freq)
{
  uint np = threadIdx.x + blockIdx.x * blockDim.x;
  uint stride = blockDim.x * gridDim.x;

  while (np < r_num_part)  
  {
    if(r_loss[np] == 0 && r_lloss[np] == 0)
    {
      double x = r_x[np];
      double y = r_y[np];
      double z = r_phi[np];
      double phimc = r_phi_avg[0];
      double xbar = r_x_avg[0];
      double ybar = r_y_avg[0];
      double epsq = r_x_sig[0] / r_y_sig[0];
      double gm = rsqrt(1.0 - r_beta*r_beta);
      double dr = scheff_sz.dr;  // fake
      double dzg = scheff_sz.dz * gm; // use dz as dzg for now.
      double wave_len = CLIGHT / r_freq;
      double hl = d_scheff_param.nz * 0.5 * dzg;
      double c2 = r_beta * gm*wave_len / TWOPI;
      double zz = 0.0, a = 0.0, cc = 0.0,
             rminsq = 0.0, rmaxsq = 0.0, b = 0.0, d = 0.0;
      int i, jm1, i1, j1, k;
      double rsq_sh = (x - xbar) * (x - xbar) / epsq + (y - ybar) * 
	    (y - ybar) * epsq;
      i = (int)(sqrt(rsq_sh) / dr + 1);
      if(i <= d_scheff_param.nx)
      {
        double ddz = z - phimc;
        while(ddz > PI || ddz < -PI)
        {
          if(ddz > 0.0) ddz -= TWOPI;
          else  ddz += TWOPI;
        }
        z = -c2 * ddz;
        if(abs(z) < hl)
        {
          zz = z + hl;
          jm1 =(int)(zz / dzg + 1);
          i1 = i + 1;
          double drdr = dr * dr;
          if(rsq_sh < 0.5 * drdr * (double)(i * i + (i - 1) * (i - 1))) 
	    i1 = i - 1;
          if(i1 < 1) i1 = 1;
          if(i1 > d_scheff_param.nx) i1 = d_scheff_param.nx;
          j1 = jm1 + 1;
          if(zz < (jm1 - 1 + 0.5) * dzg) j1 = jm1 - 1;
          if(j1 < 1) j1 = 1;
          if(j1 > d_scheff_param.nz) j1 = d_scheff_param.nz;
          if(i1 == i)
            a = 1.0;
          else
          {
            double sqrtmp = sqrt(4.0 * rsq_sh / drdr - 1.0);
            rminsq = 0.25 * drdr * (sqrtmp - 1.0) * (sqrtmp - 1.0);
            rmaxsq = rminsq + sqrtmp * drdr;
            if(i1 < i)
              a = (rmaxsq - (i - 1) * (i - 1) * drdr) / (rmaxsq - rminsq);
            else
              a = ((i1 - 1) * (i1 - 1) * drdr - rminsq) / (rmaxsq - rminsq);
          } // i1
          cc = 1.0;
          if(j1 != jm1) cc = (zz - (j1 - 1 + 0.5) * dzg) / ((jm1 - j1) * dzg);
          b = 1.0 - a;
          d = 1.0 - cc;
          k = (jm1 - 1) * d_scheff_param.nx + i;
#ifdef DOUBLE_PRECISION
          atomicAddDouble(&r_bin_tbl[k - 1], a * cc);
          k += i1 - i;
          atomicAddDouble(&r_bin_tbl[k - 1], b * cc);
          k = (j1 - 1)*d_scheff_param.nx + i;
          atomicAddDouble(&r_bin_tbl[k - 1], a * d);
          k += i1 - i;
          atomicAddDouble(&r_bin_tbl[k - 1], b * d);
#else
          atomicAdd(&r_bin_tbl[k - 1], a * cc);
          k += i1 - i;
          atomicAdd(&r_bin_tbl[k - 1], b * cc);
          k = (j1 - 1)*d_scheff_param.nx + i;
          atomicAdd(&r_bin_tbl[k - 1], a * d);
          k += i1 - i;
          atomicAdd(&r_bin_tbl[k - 1], b * d);
#endif
          // count num of particles on mesh 
          atomicAdd(&num_on_mesh, 1);
        }// if abs(z) < hl
      }// if i < nx
    }
    np += stride;
  }// if np
}

// threads in the same block share the same rp
// block : 64 * 16
// grid  : 33 + 1
// shared mem only for bin tbl
__global__
void UpdateFldTbl2Kernel(float2* r_fld_tbl2, 
#ifdef DOUBLE_PRECISION
                         double* r_bin_tbl, 
#else
                         float* r_bin_tbl,
#endif
                         double2* r_fld_tbl1)
{
  extern __shared__ float bin_smem[];
  uint fld_sz = d_scheff_param.nx * d_scheff_param.nz;
  uint readid = threadIdx.x;
  if(blockIdx.x < gridDim.x - 1)
  {
    uint rd_blkstart = blockIdx.x * fld_sz;
    while(readid < fld_sz)
    {
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nz;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nz;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = threadIdx.x % d_scheff_param.nz; // 0-63
    uint rp = blockIdx.x; // 0-32
    uint fld2_indx = zp * (d_scheff_param.nx + 1) + rp;

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      double sign = 1.0;
      uint dz;
      if(zs >= zp)
      {
        dz = zs-zp; sign = -1.0;
      }
      else
        dz = zp-zs-1; 

      uint fld1_indx = dz + rs * d_scheff_param.nz + rp * fld_sz;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += sign * bin_smem[bin_indx] * fld1_val.y; 
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
  else // last block
  {
     while(readid < fld_sz)
    {
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nx;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nx;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = d_scheff_param.nz;
    uint rp = threadIdx.x % d_scheff_param.nx; 
    uint fld2_indx = zp * (d_scheff_param.nx + 1) + rp; 

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp - zs - 1; // zp > zs for sure

      uint fld1_indx = dz + rs * d_scheff_param.nz + fld_sz * rp; 
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
    // for the last mesh point
    zp = d_scheff_param.nz;
    rp = d_scheff_param.nx;
    fld2_indx = zp * (d_scheff_param.nx + 1) + rp;
    tmpx = 0.0, tmpz = 0.0;
    uint cnt = fld_sz / blockDim.x;   
    for(int i = 0; i < cnt; ++i)
    {
      uint bin_indx = threadIdx.x + i * blockDim.x;
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp - zs - 1; // zp > zs for sure

      uint fld1_indx = dz + rs * d_scheff_param.nz + fld_sz * rp;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
}

// Every block is for one rp
// block : 64 * 16
// grid  : 33 + 1
// no use of shared mem 
__global__
void UpdateFldTbl2Kernel0(float2* r_fld_tbl2, 
#ifdef DOUBLE_PRECISION
                         double* r_bin_tbl, 
#else
                         float* r_bin_tbl,
#endif
                         double2* r_fld_tbl1)
{
//  __shared__ double2 fld1_smem[32*64];
//  __shared__ float bin_smem[32*64];
  uint fld_sz = d_scheff_param.nx * d_scheff_param.nz;
  uint readid = threadIdx.x;
  if(blockIdx.x < gridDim.x - 1)
  {
//    uint rd_blkstart = blockIdx.x * fld_sz;
//    while(readid < fld_sz)
//    {
////      fld1_smem[readid] = r_fld_tbl1[rd_blkstart + readid];
//      bin_smem[readid] = r_bin_tbl[readid];
//      readid += blockDim.x;
//    }
//    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nz;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nz;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = threadIdx.x % d_scheff_param.nz; // 0-63
    uint rp = blockIdx.x; // 0-32
    uint fld2_indx = zp * (d_scheff_param.nx + 1) + rp;

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      float bin_val = r_bin_tbl[bin_indx];
      if( bin_val == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      double sign = 1.0;
      uint dz;
      if(zs >= zp)
      {
        dz = zs-zp; sign = -1.0;
      }
      else
        dz = zp-zs-1; 

      uint fld1_indx = dz + rs*d_scheff_param.nz + rp * fld_sz;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_val * fld1_val.x;
      tmpz += sign * bin_val * fld1_val.y; 
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
  else // last block
  {
//     while(readid < fld_sz)
//    {
//      bin_smem[readid] = r_bin_tbl[readid];
//      readid += blockDim.x;
//    }
//    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nx;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nx;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = d_scheff_param.nz;
    uint rp = threadIdx.x % d_scheff_param.nx; 
    uint fld2_indx = zp * (d_scheff_param.nx + 1) + rp; 

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      float bin_val = r_bin_tbl[bin_indx];
      if(bin_val == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp - zs - 1; // zp > zs for sure

      uint fld1_indx = dz + rs * d_scheff_param.nz + fld_sz * rp; 
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_val * fld1_val.x;
      tmpz += bin_val * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
    // for the last mesh point
    zp = d_scheff_param.nz;
    rp = d_scheff_param.nx;
    fld2_indx = zp * (d_scheff_param.nx + 1) + rp;
    tmpx = 0.0, tmpz = 0.0;
    uint cnt = fld_sz / blockDim.x;   
    for(int i = 0; i < cnt; ++i)
    {
      uint bin_indx = threadIdx.x + i * blockDim.x;
      float bin_val = r_bin_tbl[bin_indx];
      if(bin_val == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp - zs - 1; // zp > zs for sure

      uint fld1_indx = dz + rs*d_scheff_param.nz + fld_sz * rp;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_val * fld1_val.x;
      tmpz += bin_val * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
}

// Every block is for one rp
// block : 64 * 16
// grid  : 33 + 1
// no use of shared mem 
__global__
void UpdateFldTbl2KernelDouble(double2* r_fld_tbl2, 
                         double* r_bin_tbl, 
                         double2* r_fld_tbl1)
{
  uint fld_sz = d_scheff_param.nx * d_scheff_param.nz;
  uint readid = threadIdx.x;
  if(blockIdx.x < gridDim.x - 1)
  {
    uint bin_block_num = blockDim.x / d_scheff_param.nz;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nz;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = threadIdx.x % d_scheff_param.nz; // 0-63
    uint rp = blockIdx.x; // 0-32
    uint fld2_indx = zp * (d_scheff_param.nx + 1) + rp;

    double tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      double bin_val = r_bin_tbl[bin_indx];
      if( bin_val == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      double sign = 1.0;
      uint dz;
      if(zs >= zp)
      {
        dz = zs-zp; sign = -1.0;
      }
      else
        dz = zp-zs-1; 

      uint fld1_indx = dz + rs*d_scheff_param.nz + rp * fld_sz;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_val * fld1_val.x;
      tmpz += sign * bin_val * fld1_val.y; 
    }
    atomicAddDouble(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAddDouble(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
  else // last block
  {
    uint bin_block_num = blockDim.x / d_scheff_param.nx;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nx;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = d_scheff_param.nz;
    uint rp = threadIdx.x % d_scheff_param.nx; 
    uint fld2_indx = zp * (d_scheff_param.nx + 1) + rp; 

    double tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      double bin_val = r_bin_tbl[bin_indx];
      if(bin_val == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp - zs - 1; // zp > zs for sure

      uint fld1_indx = dz + rs * d_scheff_param.nz + fld_sz * rp; 
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_val * fld1_val.x;
      tmpz += bin_val * fld1_val.y;
    }
    atomicAddDouble(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAddDouble(&(r_fld_tbl2[fld2_indx].y), tmpz);
    // for the last mesh point
    zp = d_scheff_param.nz;
    rp = d_scheff_param.nx;
    fld2_indx = zp * (d_scheff_param.nx + 1) + rp;
    tmpx = 0.0, tmpz = 0.0;
    uint cnt = fld_sz / blockDim.x;   
    for(int i = 0; i < cnt; ++i)
    {
      uint bin_indx = threadIdx.x + i * blockDim.x;
      double bin_val = r_bin_tbl[bin_indx];
      if(bin_val == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp - zs - 1; // zp > zs for sure

      uint fld1_indx = dz + rs * d_scheff_param.nz + fld_sz * rp;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_val * fld1_val.x;
      tmpz += bin_val * fld1_val.y;
    }
    atomicAddDouble(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAddDouble(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
}

/*! TODO Test if copy all the fld tbl into shmem for a block would save time.
 */
__global__
void ApplySpchKickKernel(double* r_x, double* r_y, double* r_phi, double* r_xp,
  double* r_yp, double* r_w, uint* r_loss, uint* r_lloss, double* r_x_avg, 
  double* r_y_avg, double* r_phi_avg, double* r_x_sig, double* r_y_sig, 
  double r_mass, uint r_num_part, double r_current, double r_freq, 
  double r_beta, uint r_adj_bunch, double r_length, 
#ifdef DOUBLE_PRECISION
  double2* r_fld_tbl, 
#else
  float2* r_fld_tbl, 
#endif
  double r_ratio_r = 1, double r_ratio_z = 1, 
  double r_ratio_q = 1, double r_ratio_gm = 1.0)
{
  uint indx = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  while(indx > 0 && indx < r_num_part) 
  {
    double x = r_x[indx];
    double y = r_y[indx];
    double xp = r_xp[indx];
    double yp = r_yp[indx];
    double phi = r_phi[indx];
    double w = r_w[indx];
    double avggamma = sqrt(1.0 / (1 - r_beta*r_beta));
    if(r_loss[indx] == 0 && r_lloss[indx] == 0) // not lost
    {
      double phimc = r_phi_avg[0];
      double xbar = r_x_avg[0];
      double ybar = r_y_avg[0];
      double epsq = r_x_sig[0] / r_y_sig[0];

      double gmg = rsqrt(1.0 - r_beta * r_beta);
      double dr = scheff_sz.dr * r_ratio_r; 
      double dzg = scheff_sz.dz*gmg * r_ratio_z;
      double wave_len = CLIGHT / r_freq;
      double hl = d_scheff_param.nz * 0.5 * dzg;
      //TODO: Where's 5721.67 from? Change this!!! 
      // 1/(2*pi^2*epsilon)= 5721.67m*mev/coul
      double c1 = 5721.67 * r_current / (r_freq * 1e6 * 
	  	(double)r_num_part * r_mass);
      double c2 = r_beta * gmg * wave_len / TWOPI;
      double c3 = r_length / (r_beta * gmg);
      double xfac = 2.0 / (epsq + 1.0); 
      double yfac = epsq * xfac;

      double bgx, bgy, bgzstar, gstar, xxbar, yybar, r, xovr,
            yovr, ddz, z, d, rod3, zod3, cbgr, cbgzs, a, b, c, 
            bgzf, er_ratio, ez_ratio;
      uint i, j, l, m;

      double gm = w / r_mass + 1.0; // gamma of one individual particle
      double bgz = sqrt((gm - 1.0) * (gm + 1.0));
      bgx = bgz * xp;
      bgy = bgz * yp;
      bgzstar = gmg * (bgz - r_beta * gm);       // momentum in bunch frame 
      gstar = gmg * (gm - r_beta * bgz);         // energy in bunch frame
      xxbar = x - xbar;
      yybar = y - ybar;
      r = sqrt(xxbar * xxbar / epsq + yybar * yybar * epsq);
      if(r == 0.0) r = 1.0e-8; // in [m]
      xovr = xxbar * xfac / r;
      yovr = yybar * yfac / r;
      ddz = phi - phimc;
      while(ddz > PI || ddz < -PI)
      {
        if(ddz > 0.0) ddz -= TWOPI;
        else ddz += TWOPI;
      }
      z = -c2 * ddz;
      if(r > d_scheff_param.nx * dr || abs(z) > hl) // not on mesh
      {
          z = -c2 * (phi - phimc);
          d = sqrt(z * z + r * r);
          double invd3 = 1.0/(d * d * d);
          rod3 = r * invd3;
          zod3 = z * invd3;
          double pl = r_beta * gmg * wave_len;
          for(int nb = 1; nb <= r_adj_bunch; ++nb)
          {
            int xi = nb;
            for(int lr = 1; lr <= 2; ++lr)
            {
              double s = z + xi * pl;
              d = sqrt(s * s + r * r);
              invd3 = 1.0/(d * d * d);
              rod3 += r * invd3;
              zod3 += s * invd3;
              xi = -xi;
            } 
          }
          // omit the part that includes the neighboring bunches
	  // num_on_mesh is count when get aa bin tbl
          double cbgcoef = c1 * c3 * PI * 0.5 * num_on_mesh; 
          cbgr = cbgcoef * rod3; 
          cbgzs = cbgcoef * zod3; 
      }
      else        // on mesh
      {
        i = 1 + (int)(r / dr);
        a = r/dr - (i - 1.0); b = 1.0 - a;
        j = 1 + (int)((z + hl) / dzg);
        c = (z + hl) / dzg - (j - 1.0); d = 1.0 - c;
        l = (j - 1) * (d_scheff_param.nx + 1) + i;
        m = l + (d_scheff_param.nx + 1);
        uint tbl_sz = (d_scheff_param.nx + 1) * (d_scheff_param.nz + 1);
        if(m >= tbl_sz)
          m = tbl_sz - 1;
        er_ratio = r_ratio_q / (r_ratio_r * r_ratio_z * r_ratio_gm * r_ratio_gm);
        ez_ratio = r_ratio_q / (r_ratio_z * r_ratio_z);
        cbgr = er_ratio * c3 * (d * (a * r_fld_tbl[l].x + b * 
	  r_fld_tbl[l - 1].x) + c * (a * r_fld_tbl[m].x + b * 
	  r_fld_tbl[m - 1].x));
        cbgzs = ez_ratio * c3 * (d * (a * r_fld_tbl[l].y + b * 
	  r_fld_tbl[l - 1].y) + c * (a * r_fld_tbl[m].y + b * 
	  r_fld_tbl[m - 1].y));
      }
      bgx += cbgr * xovr;
      bgy += cbgr * yovr;
      bgzstar += cbgzs;
      gstar = 1.0 + 0.5 * bgzstar * bgzstar;
      bgzf = gmg * (bgzstar + r_beta * gstar);
      // TODO: change this. lost, but don't know the element num
      if (bgzf <= 0.0)
        r_loss[indx] = 9999;  
      else
      {
        r_xp[indx] = bgx / bgzf;
        r_yp[indx] = bgy / bgzf;
        r_w[indx] *= 1.0 + (gm + 1.0) * (bgzf - bgz) / (gm * bgz);
      }
    }// if loss
    indx += stride;
  }// while
}

#endif

/*
// every thread would be a mesh point
// threads in the same block share the same rp
// blk (nz+1, 1, 1) -> zp
// grid (nr+1, 1, 1) -> rp
__global__
void UpdateFldTbl2Kernel_OldVersion(double2* r_fld_tbl2, 
#ifdef DOUBLE_PRECISION
                         double* r_bin_tbl, 
#else
                         float* r_bin_tbl,
#endif
                         double2* r_fld_tbl1)
{
  uint bin_indx = threadIdx.x;
  uint fld_sz = d_scheff_param.nx*d_scheff_param.nz;
  uint rd_blkstart = blockIdx.x * fld_sz;
  if(threadIdx.x < blockDim.x)
  {
    uint zp = threadIdx.x; // 0-64
    uint rp = blockIdx.x; // 0-32
    uint fld2_indx = zp*(d_scheff_param.nx+1)+rp;
    int zs, rs, dz, fld1_indx, bin_indx_tmp;
    double tmpx=0.0, tmpz=0.0, sign=1.0;
    double2 fld_val;
    double bin_val;
    #pragma unroll
    for(zs = 0; zs < d_scheff_param.nz; ++zs)
    {
      if(zs >= zp)
      {
        dz = zs-zp; sign = -1.0;
      }
      else
      {
        dz = zp-zs-1; sign = 1.0;
      }
      bin_indx_tmp = zs * d_scheff_param.nx;
      for(rs = 0; rs < d_scheff_param.nx; ++rs)
      {   
        bin_indx =  bin_indx_tmp + rs;
//        if((bin_val = bin_smem[bin_indx]) != 0.0)
        if((bin_val = r_bin_tbl[bin_indx]) != 0.0)
        {
          fld1_indx = dz + rs*d_scheff_param.nz; 
          fld_val = r_fld_tbl1[rd_blkstart+fld1_indx];
//          fld_val = fld_smem[fld1_indx];
          tmpx += bin_val * fld_val.x; 
          tmpz += sign*bin_val * fld_val.y;
        }// if
      }// for rs
    }// for zs
    r_fld_tbl2[fld2_indx].x = tmpx; 
    r_fld_tbl2[fld2_indx].y = tmpz; 
  }
}

// block : 64 * 16
// grid  : 33 + 1
// use shared mem
__global__
void UpdateFldTbl2Kernel2(float2* r_fld_tbl2, 
#ifdef DOUBLE_PRECISION
                         double* r_bin_tbl, 
#else
                         float* r_bin_tbl,
#endif
                         double2* r_fld_tbl1)
{
  __shared__ double2 fld1_smem[32*64];
  __shared__ float bin_smem[32*64];
  uint fld_sz = d_scheff_param.nx*d_scheff_param.nz;
  uint readid = threadIdx.x;
  if(blockIdx.x < gridDim.x - 1)
  {
    uint rd_blkstart = blockIdx.x * fld_sz;
    while(readid < fld_sz)
    {
      fld1_smem[readid] = r_fld_tbl1[rd_blkstart + readid];
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nz;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nz;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = threadIdx.x % d_scheff_param.nz; // 0-63
    uint rp = blockIdx.x; // 0-32
    uint fld2_indx = zp*(d_scheff_param.nx+1)+rp;

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      double sign = 1.0;
      uint dz;
      if(zs >= zp)
      {
        dz = zs-zp; sign = -1.0;
      }
      else
        dz = zp-zs-1; 

      uint fld1_indx = dz + rs*d_scheff_param.nz;
      double2 fld1_val = fld1_smem[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += sign * bin_smem[bin_indx] * fld1_val.y; 
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
  else // last block
  {
     while(readid < fld_sz)
    {
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nx;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nx;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = d_scheff_param.nz;
    uint rp = threadIdx.x % d_scheff_param.nx; 
    uint fld2_indx = zp*(d_scheff_param.nx+1)+rp; 

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp-zs-1; // zp > zs for sure

      uint fld1_indx = dz + rs*d_scheff_param.nz + fld_sz * rp; 
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
    // for the last mesh point
    zp = d_scheff_param.nz;
    rp = d_scheff_param.nx;
    fld2_indx = zp*(d_scheff_param.nx+1)+rp;
    tmpx = 0.0, tmpz = 0.0;
    uint cnt = fld_sz / blockDim.x;   
    for(int i = 0; i < cnt; ++i)
    {
      uint bin_indx = threadIdx.x + i * blockDim.x;
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp-zs-1; // zp > zs for sure

      uint fld1_indx = dz + rs*d_scheff_param.nz + fld_sz * rp;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
}
// Every two blocks for one rp 
// block : 64 * 16 
// grid  : 33 * 2 + 1
// use shared mem
__global__
void UpdateFldTbl2Kernel3(float2* r_fld_tbl2, 
#ifdef DOUBLE_PRECISION
                         double* r_bin_tbl, 
#else
                         float* r_bin_tbl,
#endif
                         double2* r_fld_tbl1)
{
  __shared__ double2 fld1_smem[32*64];
  __shared__ float bin_smem[32*64];
  uint fld_sz = d_scheff_param.nx*d_scheff_param.nz;
  uint readid = threadIdx.x;
  if(blockIdx.x < gridDim.x - 1)
  {
    uint zp = threadIdx.x % d_scheff_param.nz; // 0-63
    uint repeat_thblock_num = gridDim.x / (d_scheff_param.nx + 1);
    uint rp = blockIdx.x / repeat_thblock_num; // 0-32
    uint rd_blkstart = rp * fld_sz;
    while(readid < fld_sz)
    {
      fld1_smem[readid] = r_fld_tbl1[rd_blkstart + readid];
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num_per_thblock = blockDim.x / d_scheff_param.nz;
    uint bin_block_num = bin_block_num_per_thblock * repeat_thblock_num;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nz + 
        blockIdx.x % repeat_thblock_num * bin_block_num_per_thblock;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint fld2_indx = zp*(d_scheff_param.nx+1)+rp;

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      double sign = 1.0;
      uint dz;
      if(zs >= zp)
      {
        dz = zs-zp; sign = -1.0;
      }
      else
        dz = zp-zs-1; 

      uint fld1_indx = dz + rs*d_scheff_param.nz;
      double2 fld1_val = fld1_smem[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += sign * bin_smem[bin_indx] * fld1_val.y; 
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
  else // last block
  {
     while(readid < fld_sz)
    {
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nx;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nx;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = d_scheff_param.nz;
    uint rp = threadIdx.x % d_scheff_param.nx; 
    uint fld2_indx = zp*(d_scheff_param.nx+1)+rp; 

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp-zs-1; // zp > zs for sure

      uint fld1_indx = dz + rs*d_scheff_param.nz + fld_sz * rp; 
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
    // for the last mesh point
    zp = d_scheff_param.nz;
    rp = d_scheff_param.nx;
    fld2_indx = zp*(d_scheff_param.nx+1)+rp;
    tmpx = 0.0, tmpz = 0.0;
    uint cnt = fld_sz / blockDim.x;   
    for(int i = 0; i < cnt; ++i)
    {
      uint bin_indx = threadIdx.x + i * blockDim.x;
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp-zs-1; // zp > zs for sure

      uint fld1_indx = dz + rs*d_scheff_param.nz + fld_sz * rp;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
}
// Every two blocks for one rp 
// block : 64 * 16
// grid  : 33 * 2 + 1
// shared mem only for bin tbl
__global__
void UpdateFldTbl2Kernel4(float2* r_fld_tbl2, 
#ifdef DOUBLE_PRECISION
                         double* r_bin_tbl, 
#else
                         float* r_bin_tbl,
#endif
                         double2* r_fld_tbl1)
{
//  __shared__ double2 fld1_smem[32*64];
  extern __shared__ float bin_smem[];
  uint fld_sz = d_scheff_param.nx*d_scheff_param.nz;
  uint readid = threadIdx.x;
  if(blockIdx.x < gridDim.x - 1)
  {
    uint zp = threadIdx.x % d_scheff_param.nz; // 0-63
    uint repeat_thblock_num = gridDim.x / (d_scheff_param.nx + 1);
    uint rp = blockIdx.x / repeat_thblock_num; // 0-32
    uint rd_blkstart = rp * fld_sz;
    while(readid < fld_sz)
    {
//      fld1_smem[readid] = r_fld_tbl1[rd_blkstart + readid];
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num_per_thblock = blockDim.x / d_scheff_param.nz;
    uint bin_block_num = bin_block_num_per_thblock * repeat_thblock_num;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nz + 
        blockIdx.x % repeat_thblock_num * bin_block_num_per_thblock;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint fld2_indx = zp*(d_scheff_param.nx+1)+rp;

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      double sign = 1.0;
      uint dz;
      if(zs >= zp)
      {
        dz = zs-zp; sign = -1.0;
      }
      else
        dz = zp-zs-1; 

      uint fld1_indx = dz + rs*d_scheff_param.nz + rp*fld_sz;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += sign * bin_smem[bin_indx] * fld1_val.y; 
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
  else // last block
  {
     while(readid < fld_sz)
    {
      bin_smem[readid] = r_bin_tbl[readid];
      readid += blockDim.x;
    }
    __syncthreads();
    uint bin_block_num = blockDim.x / d_scheff_param.nx;
    uint bin_block_sz = fld_sz / bin_block_num;
    uint bin_block_id = threadIdx.x / d_scheff_param.nx;
    uint bin_start = bin_block_id * bin_block_sz;
    uint bin_end = bin_start + bin_block_sz;

    uint zp = d_scheff_param.nz;
    uint rp = threadIdx.x % d_scheff_param.nx; 
    uint fld2_indx = zp*(d_scheff_param.nx+1)+rp; 

    float tmpx = 0.0, tmpz = 0.0;
    for(uint bin_indx = bin_start; bin_indx < bin_end; ++bin_indx)
    {
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp-zs-1; // zp > zs for sure

      uint fld1_indx = dz + rs*d_scheff_param.nz + fld_sz * rp; 
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
    // for the last mesh point
    zp = d_scheff_param.nz;
    rp = d_scheff_param.nx;
    fld2_indx = zp*(d_scheff_param.nx+1)+rp;
    tmpx = 0.0, tmpz = 0.0;
    uint cnt = fld_sz / blockDim.x;   
    for(int i = 0; i < cnt; ++i)
    {
      uint bin_indx = threadIdx.x + i * blockDim.x;
      if(bin_smem[bin_indx] == 0.0)
        continue;
      uint rs = bin_indx % d_scheff_param.nx;
      uint zs = bin_indx / d_scheff_param.nx;
      uint dz = zp-zs-1; // zp > zs for sure

      uint fld1_indx = dz + rs*d_scheff_param.nz + fld_sz * rp;
      double2 fld1_val = r_fld_tbl1[fld1_indx];
      tmpx += bin_smem[bin_indx] * fld1_val.x;
      tmpz += bin_smem[bin_indx] * fld1_val.y;
    }
    atomicAdd(&(r_fld_tbl2[fld2_indx].x), tmpx);
    atomicAdd(&(r_fld_tbl2[fld2_indx].y), tmpz);
  }
}
*/
