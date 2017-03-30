#include <iostream>
#include "timer.h"

void StartTimer(cudaEvent_t* r_start, cudaEvent_t* r_stop)
{
  cudaEventCreate(r_start);
  cudaEventCreate(r_stop);
  cudaEventRecord(*r_start, 0);
}
void StopTimer(cudaEvent_t* r_start, cudaEvent_t* r_stop, std::string r_msg)
{
  cudaEventRecord(*r_stop, 0);
  cudaEventSynchronize(*r_stop);
  float elapst;
  cudaEventElapsedTime(&elapst, *r_start, *r_stop);
  std::cout << r_msg << " time: " << elapst / 1000 << " [sec]" << std::endl;
  cudaEventDestroy(*r_start);
  cudaEventDestroy(*r_stop);
}
