#ifndef BEAMLINE_EXCEPTION
#define BEAMLINE_EXCEPTION

#include <exception>

class BeamLineElementNotFoundException : public std::exception
{
public:
  const char* what() const throw()
  {
    return "Invalid beamline element name, please check it!";
  }
};

#endif
