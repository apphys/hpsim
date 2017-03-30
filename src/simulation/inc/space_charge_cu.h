#ifndef SPACE_CHARGE_KERNEL_CALL_H
#define SPACE_CHARGE_KERNEL_CALL_H

#include "space_charge_parameter.h"

void SetParameters(Beam*, uint, uint, double, uint, double);
#ifdef DOUBLE_PRECISION
void Initialize(SpaceChargeParam, double2**, double2**, double**);
void FreeMeshTables(double2*, double2*, double*);
void ResetBinTbl(double*);
void ResetFldTbl(double2*);
void UpdateTblsKernelCall(double2*, double*);
void UpdateFinalFldTblKernelCall(double2*, double*, double2*);
void KickBeamKernelCall(double2*, double r_length, double r_ratio_r = 1, 
      double r_ratio_z = 1, double r_ratio_q = 1, double r_ratio_g = 1);
#else
void Initialize(SpaceChargeParam, double2**, float2**, float**);
void FreeMeshTables(double2*, float2*, float*);
void ResetBinTbl(float*);
void ResetFldTbl(float2*);
void UpdateTblsKernelCall(double2*, float*);
void UpdateFinalFldTblKernelCall(float2*, float*, double2*);
void KickBeamKernelCall(float2*, double r_length, double r_ratio_r = 1, 
      double r_ratio_z = 1, double r_ratio_q = 1, double r_ratio_g = 1);
#endif
void UpdateMeshKernelCall();

#endif
