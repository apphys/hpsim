#include <iostream>
#include "beamline.h"
#include "utility.h"

/*!
 * \brief Contructor.
 */
BeamLine::BeamLine() : PyWrapper(),
  bl(std::vector<BeamLineElement*>())
{
}

/*!
 * \brief Destructor.
 *
 * Destroy all the elements in the list.
 */
BeamLine::~BeamLine()
{
  for(uint i = 0; i < bl.size(); ++i)
  {
    if(bl[i] != NULL)
      delete bl[i];
  }
  std::cout << "BeamLine is freed." << std::endl;
}

/*!
 * \brief Print all beamline elements.
 *
 * \callgraph
 */
void BeamLine::Print() const
{
  for(uint i = 0; i < bl.size(); ++i)
    bl[i]->Print();
}

/*!
 * \brief Print beamline elements.
 * \param r_start Name of the first element
 * \param r_end Name of the last element
 *
 * This call is inclusive. It prints all the elements in the range
 * of [r_start, r_end].
 *
 * \callgraph
 */
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

/*!
 * \brief Look up the model index of the beamline element.
 * \param r_name Name of the element
 *
 * The model index is the index of the element in the list (model)
 * that is traversed during simulation. Model index starts from zero.
 *
 * \callgraph
 */
uint BeamLine::GetElementModelIndex(std::string r_name) const 
  throw(BeamLineElementNotFoundException)
{
  for(uint i = 0; i < bl.size(); ++i)
    if (StringEqualCaseInsensitive(bl[i]->GetName(), r_name))
      return i; 
  throw BeamLineElementNotFoundException(); 
}

/*!
 * \brief Get the number of monitored elements in between two known elements
 * 	in the model.
 * \param r_begin Name of the first element
 * \param r_end Name of the last element
 *
 * This call is inclusive. It checks all elements in the range of [r_begin,
 * r_end]. This function is used in the online mode to specifiy how many 
 * output points are needed for the 2D graphics. An element can be monitored
 * by turning on the "monitor" field in the database. In the online mode, 
 * beam properties can be collected in the middle of the monitored 
 * beamline elements.
 *
 * \callgraph
 */
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

/*!
 * \brief Get the indices of the monitored elements in between two known 
 * 	elements in the model.
 * \param r_begin Name of the first element
 * \param r_end Name of the last element
 *
 * This call is inclusive. It checks all elements in the range of [r_begin,
 * r_end]. This function is used in the online mode to specifiy where to
 * show beam properties for the 2D graphics.
 *
 * \callgraph
 */
std::vector<uint> BeamLine::GetMonitoredElementsIndices(std::string r_begin, 
    std::string r_end) const
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

/*!
 * \brief Get a pointer to an element.
 * \param r_name Name of the element
 */
BeamLineElement* BeamLine::operator[](std::string r_name) 
{
  for(int i = 0; i < bl.size(); ++i)
    if(bl[i]->GetName() == r_name)
      return bl[i];
  return NULL;
}

/*!
 * \brief Get a const pointer to an element.
 * \param r_name Name of the element
 */
const BeamLineElement* BeamLine::operator[](std::string r_name) const
{
  for(int i = 0; i < bl.size(); ++i)
    if(bl[i]->GetName() == r_name)
      return bl[i];
  return NULL;
}

/*!
 * \brief Get the names of a list of elements of a certain type.
 * \param r_start Name of the first element
 * \param r_end Name of the last element
 * \param r_type Type of the element
 *
 * This call is inclusive. It checks all elements in the range of 
 * [r_start, r_end].
 */
std::vector<std::string> BeamLine::GetElementNames(std::string r_start, 
  std::string r_end, std::string r_type) const
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
