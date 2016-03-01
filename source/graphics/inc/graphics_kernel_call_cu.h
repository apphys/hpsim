#ifndef KERNEL_CALLS_H
#define KERNEL_CALLS_H

extern "C"
{
  void Set2dCurveData(uint, uint, double*, double*, float2*, uint, double2*);
  void Set2dHistogramData(uint, uint, double*, uint*, float2*, uint);
  void SetPhaseSpaceData(uint, uint, double*, double*, uint*, double*, double*, 
                         uint*, float2*, uint);
  void FindMaxMin2D(uint, uint, double*, double*, double*, double*, double4*, uint, uint* r_loss = NULL);
  void FindMaxMin1D(uint, uint, double*, uint*, double*, double4*, uint);
  void UpdateHistogram(double*, uint*, uint*, uint*, uint, uint, double, double, uint, uint);
  void TestLoss(double*, int*, uint);

  void Set3dData(uint, uint, double*, double*, double*, float4*, uint);
  void UpdateHistogram2DKernelCall(uint*, double*, double*, uint*, uint, double4*, uint, uint, uint, uint);
  void SetHistogram2DCoordinateDataKernelCall(float2*, double4*, uint, uint, uint, uint);
} // extern "C"

#endif
