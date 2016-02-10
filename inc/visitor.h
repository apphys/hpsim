#ifndef VISITOR_H
#define VISITOR_H

#include "beamline_element.h"

class Visitor
{
public:
  virtual void Visit(Buncher*) = 0;  
  virtual void Visit(Dipole*) = 0;  
  virtual void Visit(Drift*) = 0;  
  virtual void Visit(Rotation*) = 0;  
  virtual void Visit(Quad*) = 0;
  virtual void Visit(RFGap*) = 0;  
};

#endif
