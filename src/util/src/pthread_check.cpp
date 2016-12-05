#include <string>
#include <iostream>
#include <cstring>

void ThreadCheck(int r_status, std::string r_msg)
{
  if(r_status != 0)
    std::cerr << r_msg << " failure, at " << __FILE__ << ", line: " << __LINE__
              << "\nError Message: " << std::strerror(r_status)  << std::endl; 
}
