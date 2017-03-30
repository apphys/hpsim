#ifndef PLOT_DATA_H
#define PLOT_DATA_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

template<typename T>
struct PlotDataMember
{
  PlotDataMember() : size(0), d_ptr(NULL){}
  PlotDataMember(int r_sz) : size(r_sz)
  {
    cudaMalloc((void**)&d_ptr, sizeof(T)*r_sz);  
    cudaMemset(d_ptr, 0, sizeof(T)*r_sz);
  }
  ~PlotDataMember()
  {
    if(d_ptr != NULL)
      cudaFree(d_ptr);
  }
  void Resize(int r_sz)
  {
    size = r_sz;
    if(d_ptr != NULL)
      cudaFree(d_ptr);
    cudaMalloc((void**)&d_ptr, sizeof(T)*r_sz);  
    cudaMemset(d_ptr, 0, sizeof(T)*r_sz);
  }
  std::vector<T> GetValue() const
  {
    T* h_ptr = new T[size];
    cudaMemcpy(h_ptr, d_ptr, sizeof(T)*size, cudaMemcpyDeviceToHost);
    std::vector<T> rt;
    rt.assign(h_ptr, h_ptr+size);
    delete [] h_ptr;
    return rt;
  }
  T* d_ptr;
  int size;
};

/*!
 * \brief Data to be plotted in online model with 2D graphics
 */
struct PlotData
{
  PlotData();
  PlotData(int);
  void Resize(int r_sz);
  int data_size;
  PlotDataMember<double> xavg;
  PlotDataMember<double> xsig;
  PlotDataMember<double> xpavg;
  PlotDataMember<double> xpsig;
  PlotDataMember<double> xemit;
  PlotDataMember<double> yavg;
  PlotDataMember<double> ysig;
  PlotDataMember<double> ypavg;
  PlotDataMember<double> ypsig;
  PlotDataMember<double> yemit;
  PlotDataMember<double> phiavg;
  PlotDataMember<double> phisig;
  PlotDataMember<double> wavg;
  PlotDataMember<double> wsig;
  PlotDataMember<double> zemit;
  PlotDataMember<double> loss_ratio;
  PlotDataMember<double> loss_local;
};

#endif
