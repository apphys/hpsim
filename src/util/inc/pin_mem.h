#ifndef PIN_MEM_H
#define PIN_MEM_H

#include <cuda.h>
#include <cuda_runtime_api.h>

template<typename T>
void CreateDataOnPinMem(uint r_sz, T** r_pointer_d, T** r_pointer_h)
{
  cudaHostAlloc((void**)r_pointer_h, r_sz * sizeof(T), cudaHostAllocMapped);
  cudaHostGetDevicePointer((void**)r_pointer_d, *r_pointer_h, 0); 
}

template<typename T>
void FreeDataOnPinMem(T* r_pointer_h)
{
  cudaFreeHost(r_pointer_h);
}

template<typename T>
void CopyDataFromDevice(T* r_pointer_h, T* r_pointer_d, uint r_sz)
{
  cudaMemcpy(r_pointer_h, r_pointer_d, sizeof(T) * r_sz, cudaMemcpyDeviceToHost);
}

#endif
