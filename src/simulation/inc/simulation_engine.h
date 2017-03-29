#ifndef SIMULATION_ENGHINE_H
#define SIMULATION_ENGHINE_H

#include "beam.h"
#include "beamline.h"
#include "space_charge.h"
#include "simulation_parameter.h"
#include "visitor.h"
#include "py_wrapper.h"

/*! 
 * \brief Used for conducting simulations.
 *
 *  Implements visitor pattern to access BeamLineElement
 *  and strategy pattern to use different SpaceCharge methods.
 *
 */
class SimulationEngine : public Visitor, public PyWrapper
{
public:
  SimulationEngine();
  ~SimulationEngine();
  void InitEngine(Beam*, BeamLine*, SpaceCharge* r_spch = NULL, 
          bool r_graph_on = false, PlotData* r_plot_data = NULL);
  void SetSimulationParam(SimulationParam& r_param)
  {
    param_ = r_param;
  }
  void ResetEngine();
  void Simulate(std::string r_start = "", std::string r_end = "");
//  void StepSimulate(uint r_id);
  SpaceCharge* GetSpchPtr() const
  {
    return spch_;
  }
  void SetSpaceCharge(std::string r_on_off)
  {
    if(r_on_off == "on")
      param_.space_charge_on = true;
    else
      param_.space_charge_on = false;
  }

private:
  void Visit(ApertureCircular*);  
  void Visit(ApertureRectangular*);  
  void Visit(Buncher*);
  void Visit(Dipole*);
  void Visit(Diagnostics*);
  void Visit(Drift*);
  void Visit(Quad*);
  void Visit(RFGap*);
  void Visit(Rotation*);
  void Visit(SpaceChargeCompensation*);
  void Visit(Steerer*);

  //! Pointer to Beam
  Beam* beam_;
  //! Pointer to BeamLine
  BeamLine* beamline_;
  //! Pointer to SpaceCharge
  SpaceCharge* spch_;
  //! Simulation parameter
  SimulationParam param_;
  //! Model index of the last simulated beamline element, used to reset 
  int prev_end_element_index_;
};

#endif
