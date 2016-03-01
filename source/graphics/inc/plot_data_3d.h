#ifndef PLOT_DATA_3D_H
#define PLOT_DATA_3D_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

template<typename T>
std::vector<T> GetValue(T* r_d_ptr, uint r_size)
{
  T* h_ptr = new T[r_size];
  cudaMemcpy(h_ptr, r_d_ptr, sizeof(T)*r_size, cudaMemcpyDeviceToHost);
  std::vector<T> rt;
  rt.assign(h_ptr, h_ptr + r_size);
  delete [] h_ptr;
  return rt;
}

struct PlotData3D
{
  void Resize(uint r_size)
  {
    data_size = r_size;
  }
  std::vector<double> GetPhi()
  {
    return GetValue(phi, data_size);
  }
  std::vector<uint> GetLoss()
  {
    return GetValue(loss, data_size);
  }
  uint data_size;
  double* x;
  double* xp;
  double* y;
  double* yp;
  double* phi;
  double* w;
  uint* loss;
};

#endif
