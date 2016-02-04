#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <unistd.h>
#include "simulation_parameter.h"
#include "simulation_kernel.cu"
#include "beam.h"
#include "beamline.h"
#include "space_charge.h"
#include "timer.h"
#include "rfgap_parameter.h"

extern "C"
{
  namespace
  {
    SpaceCharge* spch;
    Beam* beam;
    BeamLine* bl;
    SimulationParam* param;
    uint grid_size;
    uint blck_size;
    uint cur_id = 0;
    double design_w;
    RFGapParameter* rfparam;
  }
  void SetConstOnDevice(SimulationConstOnDevice* r_const)
  {
    cudaMemcpyToSymbol(d_const, r_const, sizeof(SimulationConstOnDevice));
  }  

  void SimulateDrift(Drift* r_drift)
  {
    //std::cout << cur_id << "--- Visit Drift ---------------" << std::endl;
    int num_spch_kicks = 1;
    double length = r_drift->GetLength();
    if(param->space_charge_on)
      num_spch_kicks = std::ceil(length / spch->GetInterval());
    double hf_spch_len = 0.5 * length / num_spch_kicks;
    for(int nk = 0; nk < num_spch_kicks; ++nk)
    {
      SimulatePartialDriftKernel<<<grid_size, blck_size>>>(beam->x, beam->y, 
        beam->phi, beam->xp, beam->yp, beam->w, beam->loss, hf_spch_len, 
        r_drift->GetAperture(), cur_id);

      if(param->space_charge_on)
        spch->Start(beam, 2.0*hf_spch_len); 

      SimulatePartialDriftKernel<<<grid_size, blck_size>>>(beam->x, beam->y, 
        beam->phi, beam->xp, beam->yp, beam->w, beam->loss, hf_spch_len, 
        r_drift->GetAperture(), cur_id);
    }// for
    ++cur_id;
  }
  void SimulateQuad(Quad* r_quad)
  {
    //std::cout << cur_id << "--- Visit Quad" << std::endl;
    SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(beam->x, beam->y, beam->phi, 
      beam->xp, beam->yp, beam->w, beam->loss, r_quad->GetLength(),
      r_quad->GetAperture(), r_quad->GetGradient(), cur_id);
    if(param->space_charge_on)
      spch->Start(beam, r_quad->GetLength()); 
    SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(beam->x, beam->y, beam->phi, 
      beam->xp, beam->yp, beam->w, beam->loss, r_quad->GetLength(),
      r_quad->GetAperture(), r_quad->GetGradient(), cur_id);
    ++cur_id;
  }
  void SimulateRFGap(RFGap* r_gap)
  {
    //std::cout << cur_id << "--- Visit RFGap" << std::endl;
    // asyn copy to a device pointer made the simulation 20 times faster for DTL!
    cudaMemcpyAsync(rfparam, r_gap->GetParametersOnDevice(), sizeof(RFGapParameter), 
                    cudaMemcpyDeviceToDevice, 0);
    // check frequency change
    if(beam->freq != r_gap->GetFrequency())
    {
      double freq = r_gap->GetFrequency();
      UpdateWaveLengthKernel<<<1, 1>>>(freq);
      beam->ChangeFrequency(freq);
    }
          
    // calculate phase_in used in the gap to calculate half phase advance in the gap
    double phi_in = 0.0;
    uint bl_sz = bl->GetSize();

    // first gap in DTL tank
    if(cur_id == 1 || cur_id > 1 && (*bl)[cur_id - 2]->GetType() != "RFGap-DTL")
    {
      // make sure this is not the only gap in the tank
      if(cur_id + 2 < bl_sz && (*bl)[cur_id + 2]->GetType() == "RFGap-DTL")
      {
        RFGap* next_gap = dynamic_cast<RFGap*>((*bl)[cur_id + 2]);
        phi_in = r_gap->GetRefPhase() - 0.5 * (
          next_gap->GetRefPhase() - r_gap->GetRefPhase());
      }
      else if(cur_id + 1 >= bl_sz)
      {
        std::cerr << "BeamLine is not set up correctly. This might be "
          "the only dtl gap in the tank. There should be a quad "
          "following a DTL gap!" << std::endl;
        exit(0);
      }
      else // this is the only gap in the tank
        phi_in = r_gap->GetRefPhase();
    }
    else if (cur_id > 1) // not the first gap in DTL tank
    {
      RFGap* prev_gap = dynamic_cast<RFGap*>((*bl)[cur_id - 2]);
      phi_in = 0.5 * (prev_gap->GetRefPhase() + r_gap->GetRefPhase()); 
    }
    else // cur_id == 0
    {
      std::cerr << "A quad is missing before the DTL gap! " << std::endl;
      exit(0);
    }
    Quad* prev_quad = dynamic_cast<Quad*>((*bl)[cur_id - 1]);
    Quad* next_quad = dynamic_cast<Quad*>((*bl)[cur_id + 1]);
    double quad1_len = prev_quad->GetLength();
    double quad2_len = next_quad->GetLength();
    SimulateRFGapFirstHalfKernel<<<grid_size, blck_size/2>>>(beam->x, 
      beam->y, beam->phi, beam->xp, beam->yp, beam->w, beam->loss, 
      design_w, phi_in, rfparam, r_gap->GetLength(), quad1_len, quad2_len, false);
    // apply space charge
    if(param->space_charge_on)
      spch->Start(beam, r_gap->GetLength()); 
    SimulateRFGapSecondHalfKernel<<<grid_size, blck_size/2>>>(beam->x, 
      beam->y, beam->phi, beam->xp, beam->yp, beam->w, beam->loss, design_w, 
      phi_in, rfparam, r_gap->GetLength(), 0, quad1_len, quad2_len, false, true);
    design_w = r_gap->GetEnergyOut();
    beam->design_w = design_w;
    ++cur_id;
  }
  void SimulateRotation(Rotation*)
  {
  }
  void Cleanup()
  {
    cudaFree(rfparam); 
  }
  void Init(Beam* r_beam, BeamLine* r_bl, SpaceCharge* r_spch, SimulationParam& r_param)
  {
    spch = r_spch;
    beam = r_beam;
    bl = r_bl;
    param = &r_param; 
    grid_size = beam->grid_size;
    blck_size = beam->blck_size;
    design_w = r_beam->design_w;
    UpdateWaveLengthKernel<<<1, 1>>>(r_beam->freq);
    cudaMalloc((void**)&rfparam, sizeof(RFGapParameter));
  }

//    // start simulation
//    BeamLineElement* d_elem;
//    cudaMalloc((void**)&d_elem, sizeof(BeamLineElement));
    
////    if(r_param.space_charge_on)
////      std::cout << "Simulation start, space charge on" << std::endl;
////    if(r_param.graphics_on)
////      std::cout << "Simulation start, 2D on" << std::endl;

//    cudaEvent_t start, stop;
//    StartTimer(&start, &stop);

//    BeamLineElement* d_bl = r_bl->GetDeviceBeamLinePtr();
//    BeamLineElement* h_bl = r_bl->GetHostBeamLinePtr();
//    UpdateWaveLengthKernel<<<1, 1>>>(r_beam->freq);

//    double design_w = r_beam->design_w;
//    static int ccl_cell_num = 0; // reset to zero at the beginning of every ccl tank
//    int monitor_num = 0;
//    uint loss_num = 0;
//
//    for(uint i = 0; i <= r_end_index; ++i) 
//    {
//      double* x = r_beam->x, *xp = r_beam->xp;
//      double* y = r_beam->y, *yp = r_beam->yp;
//      double* phi = r_beam->phi, *w = r_beam->w;
//      uint* loss = r_beam->loss;
//      uint grid_size = r_beam->grid_size;
//      uint blck_size = r_beam->blck_size;

//      // SPACE CHARGE COMPENSATION
//      if(h_bl[i].type[0] == 's' && h_bl[i].type[1] == 'c' && r_spch != NULL)
//      {
//        r_spch->SetFraction(h_bl[i].t);
//      }
      
//      if(i >= r_start_index)
//      {
//        // DRIFT
//        if(h_bl[i].type[0] == 'd' && h_bl[i].type[1] == 'r' && h_bl[i].length != 0)
//        {
//          int num_spch_kicks = 1;
//          if(r_param.space_charge_on)
//            num_spch_kicks = std::ceil(h_bl[i].length/r_spch->GetInterval());
//          double hf_spch_len = 0.5*h_bl[i].length/num_spch_kicks;
//          for(int nk = 0; nk < num_spch_kicks; ++nk)
//          {
//            SimulatePartialDriftKernel<<<grid_size, blck_size>>>(x, y, phi, xp, 
//              yp, w, loss, hf_spch_len, h_bl[i].aperture1, i);
//            if(r_param.space_charge_on)
//              r_spch->Start(r_beam, 2.0*hf_spch_len); 
//
//            SimulatePartialDriftKernel<<<grid_size, blck_size>>>(x, y, phi, xp, 
//              yp, w, loss, hf_spch_len, h_bl[i].aperture1, i);
//          }// for nk
//        }
//        //STEERER
//        if(h_bl[i].type[0] == 's' && h_bl[i].type[1] == 't')
//        {
//          double blh = h_bl[i].tp;
//          double blv = h_bl[i].sp;
//          if ( blh > 1e-10 || blv > 1e-10) 
//          {
//            cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
//            std::cout << "simulate kicker" << std::endl;
//            SimulateSteererKernel<<<grid_size, blck_size>>>(xp, yp, w, loss, blh, blv);
//          }
//        }
//        // BUNCHER
//        if(h_bl[i].type[0] == 'b' && h_bl[i].type[1] == 'c' && h_bl[i].t != 0.0)
//        {
//          // check frequency change
//          if(r_beam->freq != h_bl[i].freq)
//          {
//            UpdateWaveLengthKernel<<<1, 1>>>(h_bl[i].freq);
//            r_beam->ChangeFrequency(h_bl[i].freq);
//          }
//
//          if(h_bl[i].rf_amp != 0.0)
//          {
//            cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
//            SimulateBuncherKernel<<<grid_size, blck_size>>>(x, y, phi, xp, yp, w, 
//              loss, design_w, d_elem, i);
//          }
//        }
//        // DIPOLE
//        if(h_bl[i].type[0] == 'd' && h_bl[i].type[1] == 'p')
//        {
//          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
//          SimulateFirstHalfDipoleKernel<<<grid_size, blck_size/4>>>(x, y, phi, xp,
//            yp, w, loss, d_elem);
//          if(r_param.space_charge_on)
//            r_spch->Start(r_beam, h_bl[i].length*h_bl[i].gradient); 
//          SimulateSecondHalfDipoleKernel<<<grid_size, blck_size/4>>>(x, y, phi, xp, 
//            yp, w, loss, d_elem);
//        }
//        // QUAD
//        if(h_bl[i].type[0] == 'q' && h_bl[i].type[1] == 'd')
//        {
//          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
//          SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, yp, 
//            w, loss, d_elem, i);
//          // temporary
////          r_beam->UpdateGoodParticleCount();
////          r_beam->UpdateAvgPhi(true);
////          r_beam->UpdateRelativePhi(true);
////          r_beam->UpdateSigRelativePhi(true);
////          double sigphi = r_beam->GetSigRelativePhi(true);
////          r_beam->UpdateSigX(true);
////          double sigx = r_beam->GetSigX(true);
////          r_beam->UpdateSigY(true);
////          double sigy = r_beam->GetSigY(true);
////          r_beam->UpdateAvgW(true);
////          double ke = r_beam->GetAvgW(true);
////          double gamma = (double)ke / r_beam->mass + 1.0;
////          double beta = std::sqrt(1.0-1.0/(gamma*gamma));
////          double lmbd = 299.792458/r_beam->freq;
////          double sigz = sigphi*0.5/3.14159265*beta*lmbd;
////          double sigr = std::sqrt(sigx*sigx + sigy*sigy);
////          double aratio = sigz/sigr*gamma;
////          double pratior = sigr*sigr/gamma;
////          double pratioz = sigr*sigr*sigz*gamma;
////          double cratior = sigr*sigz;
////          double cratioz = sigz*sigz*gamma*gamma;
////          std::cout << "Q" << "\t" << ke << "\t" << sigx << "\t" << sigy << "\t" << sigz << "\t" << aratio << "\t" 
////                    << pratior << "\t" << pratioz << "\t" << cratior << "\t" << cratioz <<std::endl;
//          /////////////////////////
//          if(h_bl[i].t != 0)
//            if(r_param.graphics_on)
//            {
//              r_beam->UpdateLoss();
//              double loss_local_h = r_beam->GetLossNum() - loss_num;
//              loss_num += loss_local_h;
//              loss_local_h /= r_beam->num_particle;
//              double loss_ratio_h = (double)loss_num/r_beam->num_particle;
//              cudaMemcpyAsync((r_param.plot_data->loss_local).d_ptr + monitor_num, &loss_local_h, sizeof(double), cudaMemcpyHostToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->loss_ratio).d_ptr + monitor_num, &loss_ratio_h, sizeof(double), cudaMemcpyHostToDevice, 0);
//              r_beam->UpdateEmittance();
//              cudaMemcpyAsync((r_param.plot_data->xavg).d_ptr + monitor_num, r_beam->x_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xsig).d_ptr + monitor_num, r_beam->x_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xpavg).d_ptr + monitor_num, r_beam->xp_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xpsig).d_ptr + monitor_num, r_beam->xp_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->yavg).d_ptr + monitor_num, r_beam->y_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->ysig).d_ptr + monitor_num, r_beam->y_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->ypavg).d_ptr + monitor_num, r_beam->yp_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->ypsig).d_ptr + monitor_num, r_beam->yp_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->phiavg).d_ptr + monitor_num, r_beam->phi_avg_r, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->phisig).d_ptr + monitor_num, r_beam->phi_sig_r, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->wsig).d_ptr + monitor_num, r_beam->w_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xemit).d_ptr + monitor_num, r_beam->x_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->yemit).d_ptr + monitor_num, r_beam->y_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->zemit).d_ptr + monitor_num, r_beam->z_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              double w_avg = r_beam->GetAvgW() - r_beam->GetRefEnergy();
//              cudaMemcpy((r_param.plot_data->wavg).d_ptr + monitor_num, &w_avg, sizeof(double), cudaMemcpyHostToDevice);
//              ++monitor_num;
//            } 
//          if(r_param.space_charge_on)
//            r_spch->Start(r_beam, h_bl[i].length); 
//          SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, yp, 
//            w, loss, d_elem, i);
//        } // if quad
//        // RF GAP in DTL
//        if(h_bl[i].type[0] == 'g' && h_bl[i].type[1] == 'd')
//        {
//          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
//          // check frequency change
//          if(r_beam->freq != h_bl[i].freq)
//          {
//            UpdateWaveLengthKernel<<<1, 1>>>(h_bl[i].freq);
//            r_beam->ChangeFrequency(h_bl[i].freq);
//          }
//          
//          // calculate phase_in used in the gap to calculate half phase 
//          // advance in the gap
//          double phi_in;
//          uint bl_sz = r_bl->GetBeamLineSize();
//          if(i == 1 || i > 1 && (h_bl[i-2].type[0] != 'g' || h_bl[i-2].type[1] != 'd')) // first gap in DTL tank
//          {
//            // make sure this is not the only gap in the tank
//            if(i+2 < bl_sz && h_bl[i+2].type[0] == 'g' && h_bl[i+2].type[1] == 'p')
//              phi_in = h_bl[i].phi_c - 0.5*(h_bl[i+2].phi_c-h_bl[i].phi_c); 
//            else if(i+1 >= bl_sz)
//            {
//              std::cerr << "BeamLine is not set up correctly. This might be "
//                "the only dtl gap in the tank. There should be at least a quad "
//                "following an dtl gap!" << std::endl;
//              exit(0);
//            }
//            else // this is the only gap in the tank
//              phi_in = h_bl[i].phi_c;
//          }
//          else if (i > 1) // not the first gap in DTL tank
//            phi_in = 0.5*(h_bl[i].phi_c + h_bl[i-2].phi_c);
//          else 
//          {
//            std::cerr << "A quad is missing before the DTL gap! " << std::endl;
//            exit(0);
//          }
//          
//          double quad1_len = h_bl[i-1].length;
//          double quad2_len = h_bl[i+1].length;
//          SimulateRFGapFirstHalfKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, 
//            yp, w, loss, design_w, phi_in, d_elem, quad1_len, quad2_len, false);
//          // temporary
////          r_beam->UpdateGoodParticleCount();
////          r_beam->UpdateAvgPhi(true);
////          r_beam->UpdateRelativePhi(true);
////          r_beam->UpdateSigRelativePhi(true);
////          double sigphi = r_beam->GetSigRelativePhi(true);
////          r_beam->UpdateSigX(true);
////          double sigx = r_beam->GetSigX(true);
////          r_beam->UpdateSigY(true);
////          double sigy = r_beam->GetSigY(true);
////          r_beam->UpdateAvgW(true);
////          double ke = r_beam->GetAvgW(true);
////          double gamma = (double)ke / r_beam->mass + 1.0;
////          double beta = std::sqrt(1.0-1.0/(gamma*gamma));
////          double lmbd = 299.792458/r_beam->freq;
////          double sigz = sigphi*0.5/3.14159265*beta*lmbd;
////          double sigr = std::sqrt(sigx*sigx + sigy*sigy);
////          double aratio = sigz/sigr*gamma;
////          double pratior = sigr*sigr/gamma;
////          double pratioz = sigr*sigr*sigz*gamma;
////          double cratior = sigr*sigz;
////          double cratioz = sigz*sigz*gamma*gamma;
////          std::cout << "R" << "\t" << ke << "\t" << sigx << "\t" << sigy << "\t" << sigz << "\t" << aratio << "\t" 
////                    << pratior << "\t" << pratioz << "\t" << cratior << "\t" << cratioz <<std::endl;
//          /////////////////////////
//
//          if(r_param.space_charge_on)
//            r_spch->Start(r_beam, h_bl[i].length); 
//          SimulateRFGapSecondHalfKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, 
//            yp, w, loss, design_w, phi_in, d_elem, 0, quad1_len, quad2_len, false);
//          design_w = h_bl[i].energy_out;
//          r_beam->design_w = design_w;
//        }               
//        // RF GAP in CCL
//        if(h_bl[i].type[0] == 'g' && h_bl[i].type[1] == 'c')
//        {
//          // check frequency change
//          if(r_beam->freq != h_bl[i].freq)
//          {
//            UpdateWaveLengthKernel<<<1, 1>>>(h_bl[i].freq);
//            r_beam->ChangeFrequency(h_bl[i].freq);
//          }
// 
//          // check if this is the first cell in a ccl tank
//          if( i == 0 || (h_bl[i-1].type[0] != 'g' || h_bl[i-1].type[1] != 'c'))
//            ccl_cell_num = 0;
//
//          double phi_in = 0, phi_out = 0;
//          if(ccl_cell_num == 0) // first gap in tank
//          {
//            phi_in = h_bl[i].phi_c - 0.5*(h_bl[i+1].phi_c-h_bl[i].phi_c);
//            phi_out = 0.5 * (h_bl[i+1].phi_c + h_bl[i].phi_c);
//          }
//          else if (!(h_bl[i+1].type[0] == 'g' && h_bl[i+1].type[1] == 'c')) // last gap in tank
//          {
//            phi_in = 0.5 * (h_bl[i].phi_c + h_bl[i-1].phi_c);
//            phi_out = h_bl[i].phi_c + 0.5*(h_bl[i].phi_c - h_bl[i-1].phi_c);
//          }
//          else // mid cells
//          {
//            phi_in = 0.5 * (h_bl[i].phi_c + h_bl[i-1].phi_c);
//            phi_out = 0.5 * (h_bl[i+1].phi_c + h_bl[i].phi_c);
//          }
//          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
//          int gc_blck_size = blck_size/4 > 1 ? blck_size/4 : blck_size;
//          int gc_grid_size = grid_size/2 > 1 ? grid_size/2 : grid_size;
//          SimulateRFGapFirstHalfKernel<<<grid_size, gc_blck_size>>>(x, y, phi, xp,
//            yp, w, loss, design_w, phi_in, d_elem);
//          // temporary
////          r_beam->UpdateGoodParticleCount();
////          r_beam->UpdateAvgPhi(true);
////          r_beam->UpdateRelativePhi(true);
////          r_beam->UpdateSigRelativePhi(true);
////          double sigphi = r_beam->GetSigRelativePhi(true);
////          r_beam->UpdateSigX(true);
////          double sigx = r_beam->GetSigX(true);
////          r_beam->UpdateSigY(true);
////          double sigy = r_beam->GetSigY(true);
////          r_beam->UpdateAvgW(true);
////          double ke = r_beam->GetAvgW(true);
////          double gamma = (double)ke / r_beam->mass + 1.0;
////          double beta = std::sqrt(1.0-1.0/(gamma*gamma));
////          double lmbd = 299.792458/r_beam->freq;
////          double sigz = sigphi*0.5/3.14159265*beta*lmbd;
////          double sigr = std::sqrt(sigx*sigx + sigy*sigy);
////          double aratio = sigz/sigr*gamma;
////          double pratior = sigr*sigr/gamma;
////          double pratioz = sigr*sigr*sigz*gamma;
////          double cratior = sigr*sigz;
////          double cratioz = sigz*sigz*gamma*gamma;
////          std::cout << "C" << "\t" << ke << "\t" << sigx << "\t" << sigy << "\t" << sigz << "\t" << aratio << "\t"
////                    << pratior << "\t" << pratioz << "\t" << cratior << "\t" << cratioz <<std::endl;
//          /////////////////////////
//          if(r_param.space_charge_on)
//            r_spch->Start(r_beam, h_bl[i].length);
//          SimulateRFGapSecondHalfKernel<<<grid_size, gc_blck_size>>>(x, y, phi, xp,
//            yp, w, loss, design_w, phi_out, d_elem, ccl_cell_num++);
//          design_w = h_bl[i].energy_out;
//          r_beam->design_w = design_w;
//        }
//        // ROTATION
//        if(h_bl[i].type[0] == 'r' && h_bl[i].type[1] == 'o')
//        {
//          SimulateRotationKernel<<<grid_size, blck_size>>>(x, y, xp, yp, loss, h_bl[i].phi_c);
//        }
//        // RECTANGULAR APERTURE
//        // h_bl[i].t == 0.0 when raperture is out
//        if(h_bl[i].type[0] == 'r' && h_bl[i].type[1] == 'a' && h_bl[i].t != 0.0)
//        {
//          r_beam->UpdateLoss();
//          r_beam->UpdateAvgXY();
//          SimulateRectangularApertureKernel<<<grid_size, blck_size>>>(x, y, loss, 
//            h_bl[i].aperture1, h_bl[i].tp, h_bl[i].aperture2, h_bl[i].sp, 
//            r_beam->x_avg, r_beam->y_avg, i);
//        } 
//        // CIRCULAR APERTURE
//        // h_bl[i].t == 0.0 when caperture is out
//        if(h_bl[i].type[0] == 'c' && h_bl[i].type[1] == 'a' && h_bl[i].t != 0.0)
//        {
//          r_beam->UpdateLoss();
//          r_beam->UpdateAvgXY();
//          SimulateCircularApertureKernel<<<grid_size, blck_size>>>(x, y, loss, 
//            h_bl[i].aperture1, r_beam->x_avg, r_beam->y_avg, i);
//        } 
//        // DISPLACE
//        if(h_bl[i].type[0] == 'x')
//        {
//          SimulateDisplaceKernel<<<grid_size, blck_size>>>(x, y, loss, h_bl[i].tp, h_bl[i].sp);
//        }
//        // TILT
//        if(h_bl[i].type[0] == 'z')
//        {
//          SimulateTiltKernel<<<grid_size, blck_size>>>(xp, yp, loss, h_bl[i].tp, h_bl[i].sp);
//        }
//        // DIAGNOSTICS
//        if(h_bl[i].type[0] == 'd' && h_bl[i].type[1] == 'g')
//        {
//          if(h_bl[i].t != 0)
//            if(r_param.graphics_on)
//            {
//              r_beam->UpdateLoss();
//              double loss_local_h = r_beam->GetLossNum() - loss_num;
//              loss_num += loss_local_h;
//              loss_local_h /= r_beam->num_particle;
//              double loss_ratio_h = (double)loss_num/r_beam->num_particle;
//              cudaMemcpyAsync((r_param.plot_data->loss_local).d_ptr + monitor_num, &loss_local_h, sizeof(double), cudaMemcpyHostToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->loss_ratio).d_ptr + monitor_num, &loss_ratio_h, sizeof(double), cudaMemcpyHostToDevice, 0);
//              r_beam->UpdateEmittance();
//              cudaMemcpyAsync((r_param.plot_data->xavg).d_ptr + monitor_num, r_beam->x_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xsig).d_ptr + monitor_num, r_beam->x_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xpavg).d_ptr + monitor_num, r_beam->xp_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xpsig).d_ptr + monitor_num, r_beam->xp_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->yavg).d_ptr + monitor_num, r_beam->y_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->ysig).d_ptr + monitor_num, r_beam->y_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->ypavg).d_ptr + monitor_num, r_beam->yp_avg, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->ypsig).d_ptr + monitor_num, r_beam->yp_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->phiavg).d_ptr + monitor_num, r_beam->phi_avg_r, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->phisig).d_ptr + monitor_num, r_beam->phi_sig_r, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->wsig).d_ptr + monitor_num, r_beam->w_sig, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->xemit).d_ptr + monitor_num, r_beam->x_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->yemit).d_ptr + monitor_num, r_beam->y_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              cudaMemcpyAsync((r_param.plot_data->zemit).d_ptr + monitor_num, r_beam->z_emit, sizeof(double), cudaMemcpyDeviceToDevice, 0);
//              double w_avg = r_beam->GetAvgW() - r_beam->GetRefEnergy();
//              cudaMemcpy((r_param.plot_data->wavg).d_ptr + monitor_num, &w_avg, sizeof(double), cudaMemcpyHostToDevice);
//              ++monitor_num;
//            } 
//        }
//      }// if(i >= r_start_index)
//    } // for i
    
//    StopTimer(&start, &stop, "whole Simulation");
//    r_beam->UpdateStatForPlotting();
//  }

//////////////////////////////////////////////////////////////////////////////////
/*
  void StepSimulationKernelCall(Beam* r_beam, BeamLine* r_bl, 
    SpaceCharge* r_spch, SimulationParam& r_param, uint r_index)
  {
    // start simulation
    BeamLineElement* d_elem;
    cudaMalloc((void**)&d_elem, sizeof(BeamLineElement));
    
    BeamLineElement* d_bl = r_bl->GetDeviceBeamLinePtr();
    BeamLineElement* h_bl = r_bl->GetHostBeamLinePtr();
    UpdateWaveLengthKernel<<<1, 1>>>(r_beam->freq);

    static double design_w = r_beam->GetRefEnergy();
    static int ccl_cell_num = 0;
    static uint prev_index = r_index;
    if(r_index < prev_index) // simulation restarted.
    {
      std::cout << "3D Simulation start " << std::endl;
      design_w = r_beam->GetRefEnergy();
      ccl_cell_num = 0;
    }
    prev_index = r_index;
    for(uint i = 0; i <= r_index; ++i) 
    {
      double* x = r_beam->x, *xp = r_beam->xp;
      double* y = r_beam->y, *yp = r_beam->yp;
      double* phi = r_beam->phi, *w = r_beam->w;
      uint* loss = r_beam->loss;
      uint grid_size = r_beam->grid_size;
      uint blck_size = r_beam->blck_size;
      // SPACE CHARGE COMPENSATION
      if(h_bl[i].type[0] == 's' && h_bl[i].type[1] == 'c' && r_spch != NULL)
      {
        r_spch->SetFraction(h_bl[i].t);
      }
      
      if(i == r_index)
      {
        // DRIFT
        if(h_bl[i].type[0] == 'd' && h_bl[i].type[1] == 'r' && h_bl[i].length != 0)
        {
          int num_spch_kicks = 1;
          if(r_param.space_charge_on)
            num_spch_kicks = std::ceil(h_bl[i].length/r_spch->GetInterval());
          double hf_spch_len = 0.5*h_bl[i].length/num_spch_kicks;
          for(int nk = 0; nk < num_spch_kicks; ++nk)
          {
            SimulatePartialDriftKernel<<<grid_size, blck_size>>>(x, y, phi, xp, 
              yp, w, loss, hf_spch_len, h_bl[i].aperture1, i);
            if(r_param.space_charge_on)
              r_spch->Start(r_beam, 2.0*hf_spch_len); 
            SimulatePartialDriftKernel<<<grid_size, blck_size>>>(x, y, phi, xp, 
              yp, w, loss, hf_spch_len, h_bl[i].aperture1, i);
          }// for nk
        }
        // BUNCHER
        if(h_bl[i].type[0] == 'b' && h_bl[i].type[1] == 'c' && h_bl[i].t != 0.0)
        {
          // check frequency change
          if(r_beam->freq != h_bl[i].freq)
          {
            UpdateWaveLengthKernel<<<1, 1>>>(h_bl[i].freq);
            r_beam->ChangeFrequency(h_bl[i].freq);
          }
          if(h_bl[i].rf_amp != 0.0)
          {
            cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
            SimulateBuncherKernel<<<grid_size, blck_size>>>(x, y, phi, xp, yp, w, 
              loss, design_w, d_elem, i);
          }
        }
        // DIPOLE
        if(h_bl[i].type[0] == 'd' && h_bl[i].type[1] == 'p')
        {
          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
          SimulateFirstHalfDipoleKernel<<<grid_size, blck_size/4>>>(x, y, phi, xp,
            yp, w, loss, d_elem);
          if(r_param.space_charge_on)
            r_spch->Start(r_beam, h_bl[i].length*h_bl[i].gradient); 
          SimulateSecondHalfDipoleKernel<<<grid_size, blck_size/4>>>(x, y, phi, xp, 
            yp, w, loss, d_elem);
        }
        // QUAD
        if(h_bl[i].type[0] == 'q' && h_bl[i].type[1] == 'd')
        {
          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
          SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, yp, 
            w, loss, d_elem, i);
          if(r_param.space_charge_on)
            r_spch->Start(r_beam, h_bl[i].length); 
          SimulateHalfQuadKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, yp, 
            w, loss, d_elem, i);
        } // if quad
        // RF GAP in DTL
        if(h_bl[i].type[0] == 'g' && h_bl[i].type[1] == 'd')
        {
          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
          // check frequency change
          if(r_beam->freq != h_bl[i].freq)
          {
            UpdateWaveLengthKernel<<<1, 1>>>(h_bl[i].freq);
            r_beam->ChangeFrequency(h_bl[i].freq);
          }
          
          // calculate phase_in used in the gap to calculate half phase 
          // advance in the gap
          double phi_in;
          uint bl_sz = r_bl->GetBeamLineSize();
          if(i == 1 || i > 1 && (h_bl[i-2].type[0] != 'g' || h_bl[i-2].type[1] != 'd')) // first gap in DTL tank
          {
            // make sure this is not the only gap in the tank
            if(i+2 < bl_sz && h_bl[i+2].type[0] == 'g' && h_bl[i+2].type[1] == 'p')
              phi_in = h_bl[i].phi_c - 0.5*(h_bl[i+2].phi_c-h_bl[i].phi_c); 
            else if(i+1 >= bl_sz)
            {
              std::cerr << "BeamLine is not set up correctly. This might be "
                "the only dtl gap in the tank. There should be at least a quad "
                "following an dtl gap!" << std::endl;
              exit(0);
            }
            else // this is the only gap in the tank
              phi_in = h_bl[i].phi_c;
          }
          else if (i > 1) // not the first gap in DTL tank
            phi_in = 0.5*(h_bl[i].phi_c + h_bl[i-2].phi_c);
          else 
          {
            std::cerr << "A quad is missing before the DTL gap! " << std::endl;
            exit(0);
          }
          
          double quad1_len = h_bl[i-1].length;
          double quad2_len = h_bl[i+1].length;
          SimulateRFGapFirstHalfKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, 
            yp, w, loss, design_w, phi_in, d_elem, quad1_len, quad2_len, false);
          if(r_param.space_charge_on)
            r_spch->Start(r_beam, h_bl[i].length); 
          SimulateRFGapSecondHalfKernel<<<grid_size, blck_size/2>>>(x, y, phi, xp, 
            yp, w, loss, design_w, phi_in, d_elem, 0, quad1_len, quad2_len, false);
          design_w = h_bl[i].energy_out;
        }               
        // RF GAP in CCL
        if(h_bl[i].type[0] == 'g' && h_bl[i].type[1] == 'c')
        {
          // check frequency change
          if(r_beam->freq != h_bl[i].freq)
          {
            UpdateWaveLengthKernel<<<1, 1>>>(h_bl[i].freq);
            r_beam->ChangeFrequency(h_bl[i].freq);
          }
 
          // check if this is the first cell in a ccl tank
          if( i == 0 || (h_bl[i-1].type[0] != 'g' || h_bl[i-1].type[1] != 'c'))
            ccl_cell_num = 0;

          double phi_in = 0, phi_out = 0;
          if(ccl_cell_num == 0) // first gap in tank
          {
            phi_in = h_bl[i].phi_c - 0.5*(h_bl[i+1].phi_c-h_bl[i].phi_c);
            phi_out = 0.5 * (h_bl[i+1].phi_c + h_bl[i].phi_c);
          }
          else if (!(h_bl[i+1].type[0] == 'g' && h_bl[i+1].type[1] == 'c')) // last gap in tank
          {
            phi_in = 0.5 * (h_bl[i].phi_c + h_bl[i-1].phi_c);
            phi_out = h_bl[i].phi_c + 0.5*(h_bl[i].phi_c - h_bl[i-1].phi_c);
          }
          else // mid cells
          {
            phi_in = 0.5 * (h_bl[i].phi_c + h_bl[i-1].phi_c);
            phi_out = 0.5 * (h_bl[i+1].phi_c + h_bl[i].phi_c);
          }
          cudaMemcpyAsync(d_elem, &d_bl[i], sizeof(BeamLineElement), cudaMemcpyDeviceToDevice, 0);
          int gc_blck_size = blck_size/4 > 1 ? blck_size/4 : blck_size;
          int gc_grid_size = grid_size/2 > 1 ? grid_size/2 : grid_size;
          SimulateRFGapFirstHalfKernel<<<grid_size, gc_blck_size>>>(x, y, phi, xp,
            yp, w, loss, design_w, phi_in, d_elem);
          if(r_param.space_charge_on)
            r_spch->Start(r_beam, h_bl[i].length);
          SimulateRFGapSecondHalfKernel<<<grid_size, gc_blck_size>>>(x, y, phi, xp,
            yp, w, loss, design_w, phi_out, d_elem, ccl_cell_num++);
          design_w = h_bl[i].energy_out;
        }
        // ROTATION
        if(h_bl[i].type[0] == 'r' && h_bl[i].type[1] == 'o')
        {
          SimulateRotationKernel<<<grid_size, blck_size>>>(x, y, xp, yp, loss, h_bl[i].phi_c);
        }
        // RECTANGULAR APERTURE
        // h_bl[i].t == 0.0 when raperture is out
        if(h_bl[i].type[0] == 'r' && h_bl[i].type[1] == 'a' && h_bl[i].t != 0.0)
        {
          r_beam->UpdateLoss();
          r_beam->UpdateAvgXY();
          SimulateRectangularApertureKernel<<<grid_size, blck_size>>>(x, y, loss, 
            h_bl[i].aperture1, h_bl[i].tp, h_bl[i].aperture2, h_bl[i].sp, 
            r_beam->x_avg, r_beam->y_avg, i);
        } 
        // CIRCULAR APERTURE
        // h_bl[i].t == 0.0 when caperture is out
        if(h_bl[i].type[0] == 'c' && h_bl[i].type[1] == 'a' && h_bl[i].t != 0.0)
        {
          r_beam->UpdateLoss();
          r_beam->UpdateAvgXY();
          SimulateCircularApertureKernel<<<grid_size, blck_size>>>(x, y, loss, 
            h_bl[i].aperture1, r_beam->x_avg, r_beam->y_avg, i);
        } 
        // DISPLACE
        if(h_bl[i].type[0] == 'x')
        {
          SimulateDisplaceKernel<<<grid_size, blck_size>>>(x, y, loss, h_bl[i].tp, h_bl[i].sp);
        }
        // TILT
        if(h_bl[i].type[0] == 'z')
        {
          SimulateTiltKernel<<<grid_size, blck_size>>>(xp, yp, loss, h_bl[i].tp, h_bl[i].sp);
        }
        // DIAGNOSTICS
        if(h_bl[i].type[0] == 'd' && h_bl[i].type[1] == 'g');
        // output for 3D graphics
      }// if(i == r_start_index)
    } // for i
    r_beam->UpdateStatForPlotting();
//    std::cout << "Simualtion engine: "<< r_beam->GetAvgRelativePhi() << ", " << r_beam->GetSigRelativePhi(true) <<", " << r_beam->GetAvgW(true) << ", " 
//      << r_beam->GetSigW(true) << std::endl;
  }
*/
}// extern "C"
