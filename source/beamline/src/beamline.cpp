#include <iostream>
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
  std::cout << "BeamLine is freed." << std::endl;
}

void BeamLine::Print() const
{
  for(uint i = 0; i < bl.size(); ++i)
    bl[i]->Print();
}

void BeamLine::Print(std::string r_start, std::string r_end) const
{
  int start_index = 0;
  if(r_start != "")
    start_index = GetElementModelIndex(r_start);
  int end_index = GetSize() - 1;
  if(r_end != "")
    end_index = GetElementModelIndex(r_end);
  for(uint i = start_index; i <= end_index; ++i)
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

BeamLineElement* BeamLine::operator[](std::string r_name) 
{
  for(int i = 0; i < bl.size(); ++i)
    if(bl[i]->GetName() == r_name)
      return bl[i];
  return NULL;
}
const BeamLineElement* BeamLine::operator[](std::string r_name) const
{
  for(int i = 0; i < bl.size(); ++i)
    if(bl[i]->GetName() == r_name)
      return bl[i];
  return NULL;
}

std::vector<std::string> BeamLine::GetElementNames(std::string r_start, std::string r_end, std::string r_type) const
{
  std::vector<std::string> rlt;
  int start_index = 0;
  if(r_start != "")
    start_index = GetElementModelIndex(r_start);
  int end_index = GetSize() - 1;
  if(r_end != "")
    end_index = GetElementModelIndex(r_end);
  if(r_type == "")
  {
    for(uint i = start_index; i <= end_index; ++i)
      rlt.push_back(GetElementName(i));
  }
  else
  {
    for(uint i = start_index; i <= end_index; ++i)
      if(bl[i]->GetType() == r_type)
        rlt.push_back(GetElementName(i));
  }
  return rlt;
}
