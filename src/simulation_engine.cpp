#include "simulation_engine.h"
#include "simulation_engine_cu.h"
#include "timer.h"

/*!
 * \brief Initialize the simulation setting
 * \param r_beam Beam*
 * \param r_bl   BeamLine*
 * \param r_spch SpaceCharge*
 * \param r_graph_on Indicates if it is running the online mode
 * \param r_plot_data If it is running in the online mode, data 
 *         for online plotting are written out to this pointer.
 */
void SimulationEngine::InitEngine(Beam* r_beam, BeamLine* r_bl, 
  SpaceCharge* r_spch, bool r_graph_on, PlotData* r_plot_data)
{
  beam_ = r_beam;
  beamline_ = r_bl;
  spch_ = r_spch;
  if(r_spch != NULL)
    param_.space_charge_on = true;
  else
    param_.space_charge_on = false;
  param_.graphics_on = r_graph_on;
  param_.plot_data = r_plot_data;
  SimulationConstOnDevice d_const;
  d_const.num_particle = r_beam->num_particle;
  d_const.mass = r_beam->mass;
  d_const.charge = r_beam->charge;
  SetConstOnDevice(&d_const);
}

/*!
 * \brief Simulate inclusively from an element to another.
 * \param r_start Name of the start element
 * \param r_end Name of the end element
 */
void SimulationEngine::Simulate(std::string r_start, std::string r_end)
{
  int start_index = 0;
  if(r_start != "")
    start_index = beamline_->GetElementModelIndex(r_start);
  int end_index = beamline_->GetSize() - 1;
  if(r_end != "")
    end_index = beamline_->GetElementModelIndex(r_end);
  Init(beam_, beamline_, spch_, param_);
  
  cudaEvent_t start, stop;  
  StartTimer(&start, &stop);
  for(uint i = 0; i <= end_index; ++i)
  {
    if (i >= start_index || (*beamline_)[i]->GetType() == "SpchComp")
      (*beamline_)[i]->Accept(this);
  }
  StopTimer(&start, &stop, "whole Simulation");
  Cleanup();
}

//void SimulationEngine::StepSimulate(uint r_id)
//{
//  StepSimulationKernelCall(beam_, beamline_, spch_, param_, r_id);
//}
void SimulationEngine::Visit(ApertureCircular* r_aper)
{
//  r_aper->Print();
}
void SimulationEngine::Visit(ApertureRectangular* r_aper)
{
//  r_aper->Print();
}
void SimulationEngine::Visit(Buncher* r_buncher)
{
//  r_buncher->Print();
}

void SimulationEngine::Visit(Dipole* r_dipole)
{
//  r_dipole->PrintFromDevice();
}

void SimulationEngine::Visit(Diagnostics* r_diag)
{
//  r_diag->PrintFromDevice();
}

void SimulationEngine::Visit(Drift* r_drift)
{
//  r_drift->Print();
  SimulateDrift(r_drift);
}

void SimulationEngine::Visit(Quad* r_quad)
{
//  r_quad->Print();
  SimulateQuad(r_quad);
}

void SimulationEngine::Visit(RFGap* r_gap)
{
//  r_gap->PrintFromDevice();
  if (r_gap->GetType() == "RFGap-DTL")
    SimulateDTLRFGap(r_gap);
  else if (r_gap->GetType() == "RFGap-CCL")
    SimulateCCLRFGap(r_gap);
}

void SimulationEngine::Visit(Rotation* r_rot)
{
//  r_rot->Print();
  SimulateRotation(r_rot);
}

void SimulationEngine::Visit(SpaceChargeCompensation* r_spcomp)
{
  SimulateSpaceChargeCompensation(r_spcomp);
}

void SimulationEngine::Visit(Steerer* r_steerer)
{
//  r_steerer->Print();
}
