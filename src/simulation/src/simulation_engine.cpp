#include <limits>
#include "simulation_engine.h"
#include "simulation_engine_cu.h"
#include "timer.h"

/*!
 * \brief Constructor
 */
SimulationEngine::SimulationEngine() : PyWrapper(), beam_(NULL), 
  beamline_(NULL), spch_(NULL) 
{
}

/*!
 * \brief Destructor
 *
 * \callgraph
 */
SimulationEngine::~SimulationEngine()
{
  if(beam_ != NULL)
    Cleanup();
}

/*!
 * \brief Initialize simulation setting
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
  Init(beam_, beamline_, spch_, param_);
  prev_end_element_index_ = std::numeric_limits<uint>::max();
}

/*!
 * \brief Reset some of the bookkeeping parameters when simulation 
 *	restarts.
 *
 * \callgraph
 */
void SimulationEngine::ResetEngine()
{
  Reset();
}

/*!
 * \brief Simulate inclusively from an element to another.
 * \param r_start Name of the start element
 * \param r_end Name of the end element
 *
 * \callgraph
 */
void SimulationEngine::Simulate(std::string r_start, std::string r_end)
{
  int start_index = 0;
  if(r_start != "")
    start_index = beamline_->GetElementModelIndex(r_start);
  int end_index = beamline_->GetSize() - 1;
  if(r_end != "")
    end_index = beamline_->GetElementModelIndex(r_end);
  if(start_index < prev_end_element_index_) 
    Reset();
  prev_end_element_index_ = end_index;
  cudaEvent_t start, stop;  
  StartTimer(&start, &stop);
  for(uint i = 0; i <= end_index; ++i)
    if (i >= start_index || (*beamline_)[i]->GetType() == "SpchComp")
    {
      UpdateBlIndex(i);
      (*beamline_)[i]->Accept(this);
    }
  StopTimer(&start, &stop, "Whole simulation");
  if(param_.graphics_on)
    beam_->UpdateStatForPlotting();
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(ApertureCircular* r_aper)
{
  SimulateApertureCircular(r_aper);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(ApertureRectangular* r_aper)
{
  SimulateApertureRectangular(r_aper);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(Buncher* r_buncher)
{
  SimulateBuncher(r_buncher);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(Dipole* r_dipole)
{
  SimulateDipole(r_dipole);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(Diagnostics* r_diag)
{
  SimulateDiagnostics(r_diag);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(Drift* r_drift)
{
  SimulateDrift(r_drift);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(Quad* r_quad)
{
  SimulateQuad(r_quad);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(RFGap* r_gap)
{
  if (r_gap->GetType() == "RFGap-DTL")
    SimulateDTLRFGap(r_gap);
  else if (r_gap->GetType() == "RFGap-CCL")
    SimulateCCLRFGap(r_gap);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(Rotation* r_rot)
{
  SimulateRotation(r_rot);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(SpaceChargeCompensation* r_spcomp)
{
  SimulateSpaceChargeCompensation(r_spcomp);
}

/*!
 * \brief Implementation of the visitor pattern.
 *
 * \callgraph
 */
void SimulationEngine::Visit(Steerer* r_steerer)
{
  SimulateSteerer(r_steerer);
}
