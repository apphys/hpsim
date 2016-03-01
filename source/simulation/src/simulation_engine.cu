#include <omp.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include <iterator>
#include <unistd.h>
#include "beam.h"
#include "beamline.h"
#include "space_charge.h"
#include "simulation_parameter.h"
#include "simulation_kernel.cu"
#include "rfgap_parameter.h"
#include "dipole_parameter.h"
#include "timer.h"

extern "C"
{
  namespace
  {
    SpaceCharge* spch_tmp;
    Beam* beam_tmp;
    BeamLine* bl_tmp;
    SimulationParam* param;
    uint grid_size;
    uint blck_size;
    uint cur_id;
    double design_w;
    RFGapParameter* rfparam;
    DipoleParameter* dpparam;
    uint ccl_cell_num;
    uint monitor_num;
    uint loss_num;
    void Update2DPlotData()
    {
      beam_tmp->UpdateLoss();
      double loss_local_h = beam_tmp->GetLossNum() - loss_num;
      loss_num += loss_local_h;
      loss_local_h /= beam_tmp->num_particle;
      double loss_ratio_h = (double)loss_num/beam_tmp->num_particle;
      cudaMemcpyAsync((param->plot_data->loss_local).d_ptr + monitor_num, &loss_local_h, sizeof(double), cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync((param->plot_data->loss_ratio).d_ptr + monitor_num, &loss_ratio_h, sizeof(double), cudaMemcpyHostToDevice, 0);
      beam_tmp->UpdateEmittance();
      cudaMemcpyAsync((param->plot_data->xavg).d_ptr + monitor_num, beam_tmp->x_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->xsig).d_ptr + monitor_num, beam_tmp->x_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->xpavg).d_ptr + monitor_num, beam_tmp->xp_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->xpsig).d_ptr + monitor_num, beam_tmp->xp_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->yavg).d_ptr + monitor_num, beam_tmp->y_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->ysig).d_ptr + monitor_num, beam_tmp->y_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->ypavg).d_ptr + monitor_num, beam_tmp->yp_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->ypsig).d_ptr + monitor_num, beam_tmp->yp_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->phiavg).d_ptr + monitor_num, beam_tmp->phi_avg_r, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->phisig).d_ptr + monitor_num, beam_tmp->phi_sig_r, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->wsig).d_ptr + monitor_num, beam_tmp->w_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->xemit).d_ptr + monitor_num, beam_tmp->x_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->yemit).d_ptr + monitor_num, beam_tmp->y_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      cudaMemcpyAsync((param->plot_data->zemit).d_ptr + monitor_num, beam_tmp->z_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
      double w_avg = beam_tmp->GetAvgW() - beam_tmp->GetRefEnergy();
      cudaMemcpy((param->plot_data->wavg).d_ptr + monitor_num, &w_avg, sizeof(double), cudaMemcpyHostToDevice);
      ++monitor_num;
    }
  }

  void SetConstOnDevice(SimulationConstOnDevice* r_const)
  {
    cudaMemcpyToSymbol(d_const, r_const, sizeof(SimulationConstOnDevice));
  }  

  void Cleanup()
  {
    cudaFree(rfparam); 
    cudaFree(dpparam); 
  }

  void IncreaseBlIndex()
  {
    ++cur_id;
  }

  void Reset()
  {
    cur_id = 0;
    ccl_cell_num = 0;
    monitor_num = 0;
    loss_num = 0;
    design_w = beam_tmp->design_w;
    UpdateWaveLengthKernel<<<1, 1>>>(beam_tmp->freq);
    grid_size = beam_tmp->grid_size;
    blck_size = beam_tmp->blck_size;
  }

  void Init(Beam* r_beam, BeamLine* r_bl, SpaceCharge* r_spch, SimulationParam& r_param)
  {
    spch_tmp= r_spch;
    beam_tmp = r_beam;
    bl_tmp = r_bl;
    param = &r_param; 
    cudaMalloc((void**)&rfparam, sizeof(RFGapParameter));
    cudaMalloc((void**)&dpparam, sizeof(DipoleParameter));
    Reset();
  }

  void SimulateDrift(Drift* r_drift)
  {
    //std::cout << r_drift->GetName() << ", " << cur_id << std::endl;
    int num_spch_kicks = 1;
    double length = r_drift->GetLength();
    if(param->space_charge_on)
      num_spch_kicks = std::ceil(length / spch_tmp->GetInterval());
    double hf_spch_len = 0.5 * length / num_spch_kicks;
    for(int nk = 0; nk < num_spch_kicks; ++nk)
    {
      SimulatePartialDriftKernel<<<grid_size, blck_size>>>(beam_tmp->x, beam_tmp->y, 
        beam_tmp->phi, beam_tmp->xp, beam_tmp->yp, beam_tmp->w, beam_tmp->loss, hf_spch_len, 
        r_drift->GetAperture(), cur_id);

      if(param->space_charge_on)
        spch_tmp->Start(beam_tmp, 2.0*hf_spch_len); 

      SimulatePartialDriftKernel<<<grid_size, blck_size>>>(beam_tmp->x, beam_tmp->y, 
        beam_tmp->phi, beam_tmp->xp, beam_tmp->yp, beam_tmp->w, beam_tmp->loss, hf_spch_len, 
        r_drift->GetAperture(), cur_id);
    }// for
  }

  void SimulateQuad(Quad* r_quad)
  {
    //std::cout << r_quad->GetName() << ", " << cur_id << std::endl;
    SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(beam_tmp->x, beam_tmp->y, beam_tmp->phi, 
      beam_tmp->xp, beam_tmp->yp, beam_tmp->w, beam_tmp->loss, r_quad->GetLength(),
      r_quad->GetAperture(), r_quad->GetGradient(), cur_id);
    if(param->graphics_on)
      Update2DPlotData();
    if(param->space_charge_on)
      spch_tmp->Start(beam_tmp, r_quad->GetLength()); 
    SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(beam_tmp->x, beam_tmp->y, beam_tmp->phi, 
      beam_tmp->xp, beam_tmp->yp, beam_tmp->w, beam_tmp->loss, r_quad->GetLength(),
      r_quad->GetAperture(), r_quad->GetGradient(), cur_id);
  }

  void PrepSimulateRFGap(RFGap* r_gap)
  {
    // asyn copy to a device pointer made the simulation 20 times faster for DTL!
    cudaMemcpyAsync(rfparam, r_gap->GetParametersOnDevice(), sizeof(RFGapParameter), 
                    cudaMemcpyDeviceToDevice, 0);
    // check frequency change
    if(beam_tmp->freq != r_gap->GetFrequency())
    {
      double freq = r_gap->GetFrequency(); // read from pinned mem
      UpdateWaveLengthKernel<<<1, 1>>>(freq);
      beam_tmp->ChangeFrequency(freq);
    }

  }
  void SimulateDTLRFGap(RFGap* r_gap)
  {
    PrepSimulateRFGap(r_gap);         
    // calculate phase_in used in the gap to calculate half phase advance in the gap
    double phi_in = 0.0;
    uint bl_sz = bl_tmp->GetSize();
    // if first gap in DTL tank
    if(cur_id == 1 || cur_id > 1 && (*bl_tmp)[cur_id - 2]->GetType() != "RFGap-DTL")
    {
      // make sure this is not the only gap in the tank
      if(cur_id + 2 < bl_sz && (*bl_tmp)[cur_id + 2]->GetType() == "RFGap-DTL")
      {
        RFGap* next_gap = dynamic_cast<RFGap*>((*bl_tmp)[cur_id + 2]);
        phi_in = r_gap->GetRefPhase() - 0.5 * (
          next_gap->GetRefPhase() - r_gap->GetRefPhase());
      }
      else if(cur_id + 1 >= bl_sz)
      {
        std::cerr << "BeamLine is not set up correctly. This might be "
          "the only dtl gap in the tank. There should be a quad "
          "following a DTL gap!" << std::endl;
        exit(-1);
      }
      else // this is the only gap in the tank
        phi_in = r_gap->GetRefPhase();
    }
    else if (cur_id > 1) // if not the first gap in DTL tank
    {
      RFGap* prev_gap = dynamic_cast<RFGap*>((*bl_tmp)[cur_id - 2]);
      phi_in = 0.5 * (prev_gap->GetRefPhase() + r_gap->GetRefPhase()); 
    }
    else // if cur_id == 0
    {
      std::cerr << "A quad is missing before the DTL gap! " << std::endl;
      exit(-1);
    }

    // DTL RFGap length is the cell length substracts quad lengths.
    // The lengths of the previous and following quads are needed to 
    // figure out the cell length. 
    Quad* prev_quad = dynamic_cast<Quad*>((*bl_tmp)[cur_id - 1]);
    Quad* next_quad = dynamic_cast<Quad*>((*bl_tmp)[cur_id + 1]);
    double quad1_len = prev_quad->GetLength();
    double quad2_len = next_quad->GetLength();
    SimulateRFGapFirstHalfKernel<<<grid_size, blck_size/2>>>(beam_tmp->x, 
      beam_tmp->y, beam_tmp->phi, beam_tmp->xp, beam_tmp->yp, beam_tmp->w, beam_tmp->loss, 
      design_w, phi_in, rfparam, r_gap->GetLength(), quad1_len, quad2_len, false);
    // apply space charge
    if(param->space_charge_on)
      spch_tmp->Start(beam_tmp, r_gap->GetLength()); 
    SimulateRFGapSecondHalfKernel<<<grid_size, blck_size/2>>>(beam_tmp->x, 
      beam_tmp->y, beam_tmp->phi, beam_tmp->xp, beam_tmp->yp, beam_tmp->w, beam_tmp->loss, design_w, 
      phi_in, rfparam, r_gap->GetLength(), 0, quad1_len, quad2_len, false);
    design_w = r_gap->GetEnergyOut();
    beam_tmp->design_w = design_w;
  }

  void SimulateCCLRFGap(RFGap* r_gap)
  {
    PrepSimulateRFGap(r_gap);         
  
    // check if this is the first cell in a ccl tank
    if(cur_id == 0 || (*bl_tmp)[cur_id - 1]->GetType() != "RFGap-CCL")
      ccl_cell_num = 0;

    double phi_in = 0, phi_out = 0;
    if(ccl_cell_num == 0) // first gap in tank
    {
      RFGap* next_gap = dynamic_cast<RFGap*>((*bl_tmp)[cur_id + 1]);
      double cur_phase = r_gap->GetRefPhase();
      double next_phase = next_gap->GetRefPhase();
      phi_in = cur_phase - 0.5 * (next_phase - cur_phase);
      phi_out = 0.5 * (next_phase + cur_phase);
    }
    else if ((*bl_tmp)[cur_id + 1]->GetType() != "RFGap-CCL") // last gap in tank
    {
      RFGap* prev_gap = dynamic_cast<RFGap*>((*bl_tmp)[cur_id - 1]);
      double cur_phase = r_gap->GetRefPhase();
      double prev_phase = prev_gap->GetRefPhase();
      phi_in = 0.5 * (cur_phase + prev_phase);
      phi_out = cur_phase + 0.5 * (cur_phase - prev_phase);
    }
    else // mid cells
    {
      RFGap* next_gap = dynamic_cast<RFGap*>((*bl_tmp)[cur_id + 1]);
      RFGap* prev_gap = dynamic_cast<RFGap*>((*bl_tmp)[cur_id - 1]);
      double cur_phase = r_gap->GetRefPhase();
      phi_in = 0.5 * (cur_phase + prev_gap->GetRefPhase());
      phi_out = 0.5 * (cur_phase + next_gap->GetRefPhase());
    }
//    std::cout << "phi_in = " << phi_in <<", phi_out = " << phi_out << std::endl;
    int gc_blck_size = blck_size/4 > 1 ? blck_size/4 : blck_size;
    int gc_grid_size = grid_size/2 > 1 ? grid_size/2 : grid_size;

    //cudaProfilerStart();
    SimulateRFGapFirstHalfKernel<<<grid_size, gc_blck_size>>>(beam_tmp->x, beam_tmp->y, beam_tmp->phi, beam_tmp->xp,
      beam_tmp->yp, beam_tmp->w, beam_tmp->loss, design_w, phi_in, rfparam, r_gap->GetLength());
    if(param->space_charge_on)
      spch_tmp->Start(beam_tmp, r_gap->GetLength());
    SimulateRFGapSecondHalfKernel<<<grid_size, gc_blck_size>>>(beam_tmp->x, beam_tmp->y, beam_tmp->phi, beam_tmp->xp,
      beam_tmp->yp, beam_tmp->w, beam_tmp->loss, design_w, phi_out, rfparam, r_gap->GetLength(), ccl_cell_num++);
    //cudaProfilerStop();
    design_w = r_gap->GetEnergyOut();
    beam_tmp->design_w = design_w;
  }

  void SimulateRotation(Rotation* r_rot)
  {
    //std::cout << r_rot->GetName() << ", " << cur_id << std::endl;
    SimulateRotationKernel<<<grid_size, blck_size>>>(beam_tmp->x, beam_tmp->y, beam_tmp->xp, beam_tmp->yp, beam_tmp->loss, r_rot->GetAngle());
  }
  
  void SimulateSpaceChargeCompensation(SpaceChargeCompensation* r_spcomp)
  {
    //std::cout << r_spcomp->GetName() << ", " << cur_id << std::endl;
    if(param->space_charge_on)
      spch_tmp->SetFraction(r_spcomp->GetFraction());
  }

  void SimulateApertureCircular(ApertureCircular* r_aper)
  {
    //std::cout << r_aper->GetName() << ", " << cur_id << std::endl;
    if(!r_aper->IsIn())
      return;
    beam_tmp->UpdateLoss();
    beam_tmp->UpdateAvgXY();
    SimulateCircularApertureKernel<<<grid_size, blck_size>>>(beam_tmp->x, 
        beam_tmp->y, beam_tmp->loss, r_aper->GetAperture(), 
        beam_tmp->x_avg, beam_tmp->y_avg, cur_id);
  }

  void SimulateApertureRectangular(ApertureRectangular* r_aper)
  {
    //std::cout << r_aper->GetName() << ", " << cur_id << std::endl;
    if(!r_aper->IsIn())
      return;
    beam_tmp->UpdateLoss();
    beam_tmp->UpdateAvgXY();
    SimulateRectangularApertureKernel<<<grid_size, blck_size>>>(beam_tmp->x, 
      beam_tmp->y, beam_tmp->loss, r_aper->GetApertureXLeft(), 
      r_aper->GetApertureXRight(), r_aper->GetApertureYTop(), 
      r_aper->GetApertureYBottom(),
      beam_tmp->x_avg, beam_tmp->y_avg, cur_id);
  }

  void SimulateDiagnostics(Diagnostics* r_diag)
  {
    //std::cout << r_diag->GetName() << ", " << cur_id << std::endl;
    if(param->graphics_on)
      Update2DPlotData();
  }

  void SimulateSteerer(Steerer* r_steerer)
  {
    double blh = r_steerer->GetIntegratedFieldHorizontal();
    double blv = r_steerer->GetIntegratedFieldVertical();
    if ( blh > 1e-10 || blv > 1e-10) 
      SimulateSteererKernel<<<grid_size, blck_size>>>(beam_tmp->xp, 
        beam_tmp->yp, beam_tmp->w, beam_tmp->loss, blh, blv);
  }

  void SimulateBuncher(Buncher* r_buncher)
  {
    //std::cout << r_buncher->GetName() << ", " << cur_id << std::endl;
    // check frequency change
    double bfreq = r_buncher->GetFrequency();
    if(beam_tmp->freq != bfreq) 
    {
      UpdateWaveLengthKernel<<<1, 1>>>(bfreq);
      beam_tmp->ChangeFrequency(bfreq);
    }
    if(r_buncher->IsOn())
      SimulateBuncherKernel<<<grid_size, blck_size>>>(beam_tmp->x, beam_tmp->y, 
        beam_tmp->phi, beam_tmp->xp, beam_tmp->yp, beam_tmp->w, 
        beam_tmp->loss, design_w, bfreq, r_buncher->GetVoltage(), 
        r_buncher->GetPhase(), r_buncher->GetAperture(), cur_id);
  }
  
  void SimulateDipole(Dipole* r_dipole)
  {
    //std::cout << r_dipole->GetName() << ", " << cur_id << std::endl;
    cudaMemcpyAsync(dpparam, r_dipole->GetParametersOnDevice(), sizeof(DipoleParameter), cudaMemcpyDeviceToDevice, 0);
    SimulateFirstHalfDipoleKernel<<<grid_size, blck_size/4>>>(beam_tmp->x, beam_tmp->y, beam_tmp->phi, beam_tmp->xp,
      beam_tmp->yp, beam_tmp->w, beam_tmp->loss, dpparam);
    if(param->space_charge_on)
      spch_tmp->Start(beam_tmp, r_dipole->GetRadius()*r_dipole->GetAngle()); 
    SimulateSecondHalfDipoleKernel<<<grid_size, blck_size/4>>>(beam_tmp->x, beam_tmp->y, beam_tmp->phi, beam_tmp->xp, 
      beam_tmp->yp, beam_tmp->w, beam_tmp->loss, dpparam);
  }
}
