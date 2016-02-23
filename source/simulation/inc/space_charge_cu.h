#ifndef SPACE_CHARGE_KERNEL_CALL_H
#define SPACE_CHARGE_KERNEL_CALL_H
#include "space_charge_parameter.h"

void SetParameters(Beam*, uint, uint, double, uint, double);
#ifdef DOUBLE_PRECISION
void Initialize(SpaceChargeParam, double**, double2**, float2**);
void FreeMeshTables(double*, double2*, float2*);
void ResetBinTbl(double*);
//void DistributeParticleKernelCall(double*, Beam*, double, double, uint);
void UpdateTblsKernelCall(double2*, double*);
void UpdateFinalFldTblKernelCall(float2*, double*, double2*);
#else
void Initialize(SpaceChargeParam, float**, double2**, float2**);
void FreeMeshTables(float*, double2*, float2*);
void ResetBinTbl(float*);
//void DistributeParticleKernelCall(float*, Beam*, double, double, uint);
void UpdateTblsKernelCall(double2*, float*);
void UpdateFinalFldTblKernelCall(float2*, float*, double2*);
#endif
void ResetFldTbl(float2*);
void KickBeamKernelCall(float2*, double r_length, double r_ratio_r = 1, 
      double r_ratio_z = 1, double r_ratio_q = 1, double r_ratio_g = 1);
void UpdateMeshKernelCall();
//void UpdateFldTbl1KernelCall(double2*, Beam*, double, double, uint, uint, double r_frac);

#endif
