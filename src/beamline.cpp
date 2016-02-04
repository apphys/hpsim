#include "beamline.h"
#include "utility.h"

BeamLine::BeamLine() : PyWrapper(),
  bl(std::vector<BeamLineElement*>())
{
}

BeamLine::~BeamLine()
{
  for(uint i = 0; i < bl.size(); ++i)
  {
    if(bl[i] != NULL)
      delete bl[i];
  }
}

void BeamLine::Print() const
{
  for(uint i = 0; i < bl.size(); ++i)
    bl[i]->Print();
}

uint BeamLine::GetElementModelIndex(std::string r_name) const throw(BeamLineElementNotFoundException)
{
  for(uint i = 0; i < bl.size(); ++i)
    if (StringEqualCaseInsensitive(bl[i]->GetName(), r_name))
      return i; 
  throw BeamLineElementNotFoundException(); 
}

uint BeamLine::GetNumOfMonitors(std::string r_begin, std::string r_end) const
{
  uint idb = 0;
  if (r_begin != "")
    idb = GetElementModelIndex(r_begin);
  uint ide = bl.size() - 1;
  if (r_end != "")
    ide = GetElementModelIndex(r_end);
  uint cnt = 0;
  for (uint i = idb; i <= ide; ++i)
    if (bl[i]->IsMonitorOn()) 
      ++cnt;
  return cnt;
}       
std::vector<uint> BeamLine::GetMonitoredElementsIndices(std::string r_begin, std::string r_end) const
{
  uint idb = 0;
  if (r_begin != "")
    idb = GetElementModelIndex(r_begin);
  uint ide = bl.size() - 1;
  if (r_end != "")
    ide = GetElementModelIndex(r_end);
  std::vector<uint> indices;
  for (uint i = idb; i <= ide; ++i)
    if (bl[i]->IsMonitorOn()) 
      indices.push_back(i);
  return indices;
}
