#ifndef BEAM_CU_H
#define BEAM_CU_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include "beam.h"
 
void CreateBeamOnDevice(Beam* r_beam);
void CreatePartialsOnDevice(Beam* r_beam, uint r_grid_size, uint r_blck_size);
void FreePartialsOnDevice(Beam* r_beam);
void UpdateBeamOnDevice(Beam* r_beam, double* r_x_h, double* r_xp_h, 
  double* r_y_h, double* r_yp_h, double* r_phi_h, double* r_w_h, 
  uint* r_loss_h = NULL, uint* r_lloss_h = NULL);
void FreeBeamOnDevice(Beam* r_beam);
void CopyBeamFromDevice(Beam* r_beam, double* r_x_h, double* r_xp_h,
  double* r_y_h, double* r_yp_h, double* r_phi_h, double* r_w_h,
  uint* r_loss_h, uint* r_lloss_h, uint* r_num_loss_h);
void CopyParticleFromDevice(Beam* r_beam, uint r_index, double* r_x_h, 
  double* r_xp_h, double* r_y_h, double* r_yp_h, double* r_phi_h, double* r_w_h,
  uint* r_loss_h, uint* r_lloss_h);
void UpdateLossCountKernelCall(Beam* r_beam, bool r_lloss = false);
void UpdateLongitudinalLossCoordinateKernelCall(Beam* r_beam);
void UpdateAvgOfOneVariableKernelCall(Beam* r_beam, double* r_x, 
  double* r_x_avg, bool r_good_only = false);
void UpdateSigmaOfOneVariableKernelCall(Beam* r_beam, double* r_x, 
  double* r_x_sig, bool r_good_only = false);
//void UpdateSigmaR(Beam* r_beam);
void UpdateHorizontalEmittanceKernelCall(Beam* r_beam);
void UpdateVerticalEmittanceKernelCall(Beam* r_beam);
void UpdateLongitudinalEmittanceKernelCall(Beam* r_beam);
void UpdateAvgSigXYKernelCall(Beam* r_beam);
void UpdateAvgXYKernelCall(Beam* r_beam);
void UpdateMaxRPhiKernelCall(Beam* r_beam);
void UpdateRelativePhiKernelCall(Beam* r_beam, bool r_use_good = false);
void SetDoubleValue(double* r_arr, uint r_index, double r_val);
void CutBeamKernelCall(double* r_coord, uint* r_loss, double r_min, 
  double r_max, uint r_num, uint r_grid, uint r_blck);
void ShiftVariableKernelCall(Beam* r_beam, double* r_var, double r_val);
void ChangeFrequnecyKernelCall(Beam* r_beam, double r_freq_ratio);
void UpdateGoodParticleCountKernelCall(Beam* r_beam);
template<typename T>
void CopyVariable(T** r_des, T** r_source)
{
  cudaMemcpy(*r_des, *r_source, sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
T GetDataFromDevice(T* r_arr, uint r_index)
{
  T rtl;
  cudaMemcpy(&rtl, r_arr+r_index, sizeof(T), cudaMemcpyDeviceToHost);
  return rtl; 
}

template<typename T>
std::vector<T> GetArrayFromDevice(T* r_arr, uint r_num)
{
  T* rt_arr = new T[r_num]; 
  cudaMemcpy(rt_arr, r_arr, sizeof(T)*r_num, cudaMemcpyDeviceToHost);
  std::vector<T> rt;
  rt.assign(rt_arr, rt_arr+r_num);
  delete [] rt_arr;
  return rt; 
}

template<typename T>
void CopyArrayFromDevice(T* r_src, T* r_out, uint r_num)
{
  cudaMemcpy(r_out, r_src, sizeof(T)*r_num, cudaMemcpyDeviceToHost);
}
#endif
