#ifndef TIMER_H
#define TIMER_H

#include <cuda.h>
#include <cuda_runtime_api.h>

void StartTimer(cudaEvent_t* r_start, cudaEvent_t* r_stop);
void StopTimer(cudaEvent_t* r_start, cudaEvent_t* r_stop, std::string r_msg);

#endif
