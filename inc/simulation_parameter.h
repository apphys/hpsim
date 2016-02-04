#ifndef SIMULATION_PARAMETER_H
#define SIMULATION_PARAMETER_H

#include "plot_data.h"

struct SimulationParam
{
  bool space_charge_on;
  bool graphics_on;
  bool graphics3D_on;
  PlotData* plot_data;
};

struct SimulationConstOnDevice
{
  uint num_particle;
  double mass;
  double charge;
};

#endif
