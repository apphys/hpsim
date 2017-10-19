#ifndef EPICSPUT_H
#define EPICSPUT_H

#include <string>
#include <vector>
#include "plot_data.h"

extern bool epics_put_stop_flag;
struct EPICSPutArg
{
  PlotData* data;
};

void EPICSPut(std::vector<std::string>&, std::vector<std::vector<double> >&);

#endif
