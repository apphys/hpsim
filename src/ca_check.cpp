#include <string>
#include <iostream>
#include "cadef.h"

bool CACheck(int r_status, std::string r_op, std::string r_pv)
{
  if(r_status != ECA_NORMAL)
  {
    std::cerr << "CA Error: ";
    if(r_pv.compare("") != 0)
      std::cerr << r_op << " failure for PV: " << r_pv << std::endl;
    else
      std::cerr << r_op << " failure" << std::endl; 
    return false;
  }
  return true;
}
