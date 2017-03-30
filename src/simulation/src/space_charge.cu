#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cmath>
#include "space_charge_kernel.cu"
#include "space_charge_parameter.h"
#include "beam.h"
#include "timer.h"

#ifdef _DEBUG
#include <iostream>
#include <fstream>
#include <iomanip>
#endif

namespace
{
  uint nr;
  uint nz;
  Beam* beam;
  double beta;
  double freq;
  double frac;
  uint adj_bunch;
}

void Initialize(SpaceChargeParam r_param, 
		double2** r_fld_tbl1, 
#ifdef DOUBLE_PRECISION
		double2** r_fld_tbl2,
		double** r_bin_tbl
#else
		float2** r_fld_tbl2,
		float** r_bin_tbl 
#endif
)
{
  nr = r_param.nx;
  nz = r_param.nz;
  cudaMemcpyToSymbol(d_scheff_param, &r_param, sizeof(SpaceChargeParam));
#ifdef DOUBLE_PRECISION
  uint sz = nr * nz * sizeof(double);
#else
  uint sz = nr * nz * sizeof(float);
#endif
  cudaMalloc((void**)r_bin_tbl, sz);
  cudaMemset(*r_bin_tbl, 0.0, sz);

  sz = nr * nz * (nr + 1) * sizeof(double2);
  cudaMalloc((void**)r_fld_tbl1, sz);
  cudaMemset(*r_fld_tbl1, 0.0, sz);

#ifdef DOUBLE_PRECISION
  sz = (nz + 1) * (nr + 1) * sizeof(double2);
#else
  sz = (nz + 1) * (nr + 1) * sizeof(float2);
#endif
  cudaMalloc((void**)r_fld_tbl2, sz);
  cudaMemset(*r_fld_tbl2, 0.0, sz);
}

void FreeMeshTables(double2* r_fld_tbl1, 
#ifdef DOUBLE_PRECISION
		    double2* r_fld_tbl2,
		    double* r_bin_tbl
#else
		    float2* r_fld_tbl2,
		    float* r_bin_tbl 
#endif
)
{
  cudaFree(r_bin_tbl);
  cudaFree(r_fld_tbl1);
  cudaFree(r_fld_tbl2);
}

void SetParameters(Beam* r_beam, uint r_nr, uint r_nz, double r_frac, 
  uint r_adj_bunch, double r_beta)
{
  beam = r_beam;
  freq = beam->freq;
  beta = r_beta;
  frac = r_frac;
  adj_bunch = r_adj_bunch;
  nr = r_nr; nz = r_nz;
}

void ResetBinTbl(
#ifdef DOUBLE_PRECISION
		 double* r_bin_tbl
#else
		 float* r_bin_tbl
#endif
)
{
#ifdef DOUBLE_PRECISION
  uint sz = nr*nz*sizeof(double);
#else
  uint sz = nr*nz*sizeof(float);
#endif
  cudaMemset(r_bin_tbl, 0.0, sz);
}

void ResetFldTbl(
#ifdef DOUBLE_PRECISION
		 double2* r_fld_tbl2
#else
		 float2* r_fld_tbl2
#endif
)
{
#ifdef DOUBLE_PRECISION
  uint sz = (nz + 1) * (nr + 1) *  sizeof(double2);
#else
  uint sz = (nz + 1) * (nr + 1) * sizeof(float2);
#endif
  cudaMemset(r_fld_tbl2, 0.0, sz);
}

//TODO: lauching a kernel with just one thread isn't the best way to do
// it. Since the overhead is small, itgnore this for now.
void UpdateMeshKernelCall()
{
  if(adj_bunch != 0)
  {
    beam->UpdateMaxRPhi();
    UpdateScheffMeshSizeKernel<<<1, 1>>>(beam->r_max, NULL, freq, beta);
  }
  else
  {
    if(beam->x_sig_good > beam->y_sig_good)
      UpdateScheffMeshSizeKernel<<<1, 1>>>(beam->x_sig_good, beam->phi_sig_good,
	freq, beta, 3.0);
    else
      UpdateScheffMeshSizeKernel<<<1, 1>>>(beam->y_sig_good, beam->phi_sig_good,
	freq, beta, 3.0);
  }
}

/*!
 * \brief Update the charge density table & the intermediate Green's function
 *	table.
 *
 * Tried to overlap with streams, but seen almost no benefit on Fermi GPUs.
 * 
 * \callgraph
 */
void UpdateTblsKernelCall(double2* r_fld_tbl1, 
#ifdef DOUBLE_PRECISION
			  double* r_bin_tbl
#else
			  float* r_bin_tbl
#endif
)
{
  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  // Update the first field table, r_nz must be smaller than 512, 2^n
  uint blkx = nz;
  uint blky = (512 / nz < nr) ? 512 / nz : nr;
  uint gridx = nr / blky;
  uint gridy = nr + 1;
  // 4 = 32/8 (rs), 33(rp) is the receiving point r.
  dim3 grid(gridx, gridy, 1); 
  // 32(nr) source point r, 8 is to make sure thread num < 1024
  dim3 blk(blkx, blky, 1);  
  //cudaProfilerStart();
  UpdateFldTbl1Kernel<<<grid, blk, 0, stream1>>>(r_fld_tbl1, freq, beta, 
    beam->num_particle, beam->current*frac/beam->mass, adj_bunch);
  //cudaProfilerStop();
  // update the bin table
  InitNumOnMesh<<<1, 1, 0, stream2>>>();
  uint grid_size = beam->grid_size;
  uint blck_size = beam->blck_size;
  //cudaProfilerStart();
  cudaEvent_t start, stop;
  //StartTimer(&start, &stop);
  DistributeParticleKernel<<<grid_size, blck_size / 4, 0, stream2>>>(
    r_bin_tbl, beam->x, beam->y, beam->phi_r, beam->loss, beam->lloss, 
    beam->x_avg_good, beam->y_avg_good, beam->phi_avg_r, beam->x_sig_good, 
    beam->y_sig_good, beam->num_particle, beta, freq);
  //StopTimer(&start, &stop, "DistributeParticleKernel: ");
  //cudaProfilerStop();
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
}

void UpdateFinalFldTblKernelCall(
#ifdef DOUBLE_PRECISION
				 double2* r_fld_tbl2,
				 double* r_bin_tbl, 
#else
				 float2* r_fld_tbl2, 
				 float* r_bin_tbl, 
#endif
				 double2* r_fld_tbl1)
{
  uint blk = nz + 1;
  uint grid = nr + 1;
  //cudaProfilerStart();
#ifdef DOUBLE_PRECISION
  UpdateFldTbl2KernelDouble<<<nr + 2, 1024>>>(r_fld_tbl2, r_bin_tbl, r_fld_tbl1);
#else
  if(nr * nz <= 12 * 1024)
    UpdateFldTbl2Kernel<<<nr + 2, 1024, nr * nz * sizeof(float)>>>(r_fld_tbl2, 
      r_bin_tbl, r_fld_tbl1);
  else
    UpdateFldTbl2Kernel0<<<nr + 2, 1024>>>(r_fld_tbl2, r_bin_tbl, r_fld_tbl1);
#endif
  //cudaProfilerStop();

#ifdef _DEBUG
  cudaThreadSynchronize();
  std::cout << "done final fld tbl" << std::endl;  
  PrintFinalFldTbl(r_fld_tbl2);
#endif
}

void KickBeamKernelCall(
#ifdef DOUBLE_PRECISION
			double2* r_fld_tbl2,
#else
			float2* r_fld_tbl2,
#endif
			double r_length, double r_ratio_r, double r_ratio_z, 
			double r_ratio_q, double r_ratio_g
) 
{
  uint grid_size = beam->grid_size;
  uint blck_size = beam->blck_size / 4;
  //cudaProfilerStart();
  ApplySpchKickKernel<<<grid_size, blck_size>>>(beam->x, beam->y, 
    beam->phi_r, beam->xp, beam->yp, beam->w, beam->loss, beam->lloss, 
    beam->x_avg_good, beam->y_avg_good, beam->phi_avg_r, beam->x_sig_good, 
    beam->y_sig_good, beam->mass, beam->num_particle, beam->current*frac, 
    freq, beta, adj_bunch, r_length, r_fld_tbl2, r_ratio_r, r_ratio_z, 
    r_ratio_q, r_ratio_g);
  //cudaProfilerStop();
#ifdef _DEBUG
  cudaThreadSynchronize();
  std::cout << "done kick" << std::endl;  
#endif
}

#ifdef _DEBUG
void PrintFldTbl1(double2* r_fld_tbl1, uint r_nr, uint r_nz)
{
  uint sz = r_nr * r_nz * (r_nr + 1);
  double2* h_fld_tbl1 = new double2[sz];
  cudaMemcpy(h_fld_tbl1, r_fld_tbl1, sz*sizeof(double2), cudaMemcpyDeviceToHost);
  std::ofstream tout("fld_tbl1.dat");
  tout << std::setprecision(16);
  // Used to compare with CPU code
  for(int i = 0; i < r_nr; ++i)  // rs
    for(int j = 0; j < r_nz; ++j)  // zp
      for(int k = 0; k < r_nr+1; ++k)  // rp
      {
        double2 tmp = h_fld_tbl1[j + i * r_nz + k * r_nz * r_nr];
        tout << i << "\t" << j << "\t" << k << "\t" << tmp.x << "\t" << tmp.y 
	  << std::endl;
      }
// the real order of the table
/*
  for(int i = 0; i < sz; ++i)
  {
    double2 tmp = h_fld_tbl1[i];
    tout << tmp.x << "\t" << tmp.y << std::endl;
  }
*/
  tout.close();
  delete [] h_fld_tbl1;
}
#endif

#ifdef _DEBUG
void PrintBinTbl(float* r_tbl, uint sz, std::string r_file)
{
  float* h_tbl = new float[sz];
  cudaMemcpy(h_tbl, r_tbl, sz*sizeof(float), cudaMemcpyDeviceToHost);
  std::ofstream tout(r_file.c_str());
  tout << std::setprecision(16);
  for(int i = 0; i < sz; ++i)
    tout << i << "\t" << h_tbl[i] << std::endl;

  tout.close();
  delete [] h_tbl;
}
#endif

#ifdef _DEBUG
void PrintFinalFldTbl(float2* r_tbl)
{
  uint sz = (nr + 1) * (nz + 1);
  float2* h_tbl = new float2[sz];
  cudaMemcpy(h_tbl, r_tbl, sz * sizeof(float2), cudaMemcpyDeviceToHost);
  std::ofstream tout("final_fld_tbl.dat");
  tout << std::setprecision(10);
  for(int i = 0; i < sz; ++i)
    tout << i << "\t" << h_tbl[i].x << "\t" << h_tbl[i].y << std::endl;
  tout.close();
  delete [] h_tbl;
}
#endif
