#ifndef CONSTANT_H
#define CONSTANT_H

#include <cuda.h>
#include <cuda_runtime_api.h>

__constant__ double CLIGHT = 299.792458;
__constant__ double PI = 3.14159265358979323846;
__constant__ double TWOPI = 3.14159265358979323846*2.0;
__constant__ double NIFN = -2147483647;
__constant__ double RADIAN = 3.14159265358979323846/180.0;

#endif
