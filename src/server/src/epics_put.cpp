#include <errno.h> //for EBUSY 
#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cadef.h>
#include "ca_check.h"
#include "pthread_check.h"
#include "tool_lib.h"
#include "epics_put.h"

bool epics_put_stop_flag = false;
void EPICSPut(std::vector<std::string>& r_pv_name, 
  std::vector<std::vector<double> >& r_val)
{
  CACheck(ca_context_create(ca_disable_preemptive_callback),
    "ServerRoutine:ca_context_create"); 
  
  const int num_channels = r_pv_name.size();
  pv* pvs = new pv[num_channels];
  for (int i = 0; i < num_channels; ++i)
    pvs[i].name = const_cast<char*>(r_pv_name[i].c_str());
  connect_pvs(pvs, num_channels);
  
  for (int i = 0; i < num_channels; ++i)
  {
    const int pv_value_length = r_val[i].size();
    double* vals = new double[pv_value_length];
    std::copy(r_val[i].begin(), r_val[i].end(), vals);
    ca_array_put(DBR_DOUBLE, pv_value_length, pvs[i].chid, vals);
    delete [] vals;
  }
  int result = ca_pend_io(caTimeout);
  if (result == ECA_TIMEOUT) {
      fprintf(stderr, "Write operation timed out: Data was not written.\n");
  }
  ca_context_destroy();
}
