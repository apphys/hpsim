#ifndef BEAMLINE_H
#define BEAMLINE_H

#include <vector>
#include "beamline_element.h"
#include "beamline_exception.h"
#include "py_wrapper.h"

typedef unsigned int uint;

struct BeamLine : public PyWrapper
{
  BeamLine();
  ~BeamLine();
  void AddElement(BeamLineElement*);
  uint GetSize() const;
  uint GetElementModelIndex(std::string) const throw(BeamLineElementNotFoundException);
  std::string GetElementName(uint) const;
  uint GetNumOfMonitors(std::string r_begin = "", std::string r_end = "") const; // inclusive
  std::vector<uint> GetMonitoredElementsIndices(std::string r_begin = "", std::string r_end = "") const;
  void Print() const;
  const BeamLineElement* operator[](uint) const;
  BeamLineElement* operator[](uint);
  const BeamLineElement* operator[](std::string) const;
  BeamLineElement* operator[](std::string);
   
  std::vector<BeamLineElement*> bl;
};
inline
uint BeamLine::GetSize() const
{
  return bl.size();
}
inline
void BeamLine::AddElement(BeamLineElement* r_elem)
{
  bl.push_back(r_elem);
}
inline
std::string BeamLine::GetElementName(uint r_index) const
{
  return bl[r_index]->GetName();  
}

inline
const BeamLineElement* BeamLine::operator[](uint r_index) const
{
  return bl[r_index];
}

inline
BeamLineElement* BeamLine::operator[](uint r_index)
{
  return bl[r_index];
}
#endif
