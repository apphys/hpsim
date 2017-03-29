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
  uint GetElementModelIndex(std::string) const 
      throw(BeamLineElementNotFoundException);
  std::string GetElementName(uint) const;
  std::vector<std::string> GetElementNames(std::string r_begin = "", 
      std::string r_end = "", std::string r_type = "") const;
  uint GetNumOfMonitors(std::string r_begin = "", std::string r_end = "") const;
  std::vector<uint> GetMonitoredElementsIndices(std::string r_begin = "", 
      std::string r_end = "") const;
  void Print() const;
  void Print(std::string, std::string) const;
  const BeamLineElement* operator[](uint) const;
  BeamLineElement* operator[](uint);
  const BeamLineElement* operator[](std::string) const;
  BeamLineElement* operator[](std::string);
   
  //! List of BeamLineElements. It sits on the host.
  std::vector<BeamLineElement*> bl;
};

/*!
 * \brief Get the total number of elements in the model.
 */
inline
uint BeamLine::GetSize() const
{
  return bl.size();
}

/*!
 * \brief Add an element to the model.
 * \param r_elem A pointer to the new element
 *
 * The elements will be destroyed by BeamLine when the destructor is called.
 */
inline
void BeamLine::AddElement(BeamLineElement* r_elem)
{
  bl.push_back(r_elem);
}

/*!
 * \brief Get the name of an element with a given model index.
 * \param r_index Model index of the element
 */
inline
std::string BeamLine::GetElementName(uint r_index) const
{
  return bl[r_index]->GetName();  
}

/*!
 * \brief Get a const pointer to an element.
 * \param r_index Model index of the element
 */
inline
const BeamLineElement* BeamLine::operator[](uint r_index) const
{
  return bl[r_index];
}

/*!
 * \brief Get a pointer to an element.
 * \param r_index Model index of the element
 */
inline
BeamLineElement* BeamLine::operator[](uint r_index)
{
  return bl[r_index];
}

#endif
