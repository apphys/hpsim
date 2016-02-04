#include <iostream>
#include "beamline_element.h"
#include "pin_mem.h"
#include "visitor.h"

BeamLineElement::BeamLineElement(std::string r_name, std::string r_type, 
  double r_length) : name_(r_name), type_(r_type), length_(r_length), 
  aperture_(0.0), monitor_on_(false)
{
}

Drift::Drift(std::string r_name) : BeamLineElement(r_name, "Drift")
{
}

void Drift::Print() const
{
  std::cout << GetName() << ": " << GetType() << ", length = " << GetLength() 
    << ", aper " << GetAperture() << std::endl;
}

void Drift::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

Rotation::Rotation(std::string r_name) : BeamLineElement(r_name, "Rotation")
{
}

void Rotation::Print() const
{
  std::cout << GetName() << ": " << GetType() << ", angle = "   
    << GetAngle() << std::endl;
}

void Rotation::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

Quad::Quad(std::string r_name) : BeamLineElement(r_name, "Quad"),
  gradient_(0.0)
{
}

void Quad::Print() const
{
  std::cout << GetName() << ": " << GetType() << ", length = " << GetLength() << ", aper "
    << GetAperture() << ", gradient = " << GetGradient()
    << ", monitor = " << (IsMonitorOn()? "on" : "off")<< std::endl;
}

void Quad::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

RFGap::RFGap(std::string r_name, std::string r_type) : 
  BeamLineElement(r_name, r_type)
{
  CreateDataOnPinMem(1, &param_h_, &param_d_);
}

RFGap::~RFGap() 
{
  if (param_h_ != NULL)
    FreeDataOnPinMem(param_h_); 
}

void RFGap::Sync()
{
  sync();  
}
void RFGap::Print() const
{
  std::cout << GetName() + ": " << GetType() << ", length = " << GetLength() 
    << ", aper " << GetAperture() << ", freq = " << param_h_[0].frequency << ", \n"
    << "\t cell_len_bl = " << param_h_[0].cell_length_over_beta_lambda 
    << ", amp = " << param_h_[0].amplitude 
    << ", ref_phi = " << param_h_[0].phase_ref << ", \n"
    << "\t beam_phi_shift = " << param_h_[0].phase_shift
    << ", energy_out = " << param_h_[0].energy_out 
    << ", beta_center = " << param_h_[0].beta_center << ", \n"
    << "\t dg = " << param_h_[0].dg << ", t = " << param_h_[0].t 
    << ", tp = " << param_h_[0].tp << ", sp = " << param_h_[0].sp << ", \n"
    << "\t fit_betac = " << param_h_[0].fit_beta_center 
    << ", fit_betamin = " << param_h_[0].fit_beta_min 
    << ", fit_t0 = " << param_h_[0].fit_t0 << ", \n"
    << "\t fit_t1 = " << param_h_[0].fit_t1
    << ", fit_t2 = " << param_h_[0].fit_t2
    << ", fit_t3 = " << param_h_[0].fit_t3 << ", \n"
    << "\t fit_t4 = " << param_h_[0].fit_t4
    << ", fit_t5 = " << param_h_[0].fit_t5
    << ", fit_s1 = " << param_h_[0].fit_s1 << ", \n"
    << "\t fit_s2 = " << param_h_[0].fit_s2
    << ", fit_s3 = " << param_h_[0].fit_s3
    << ", fit_s4 = " << param_h_[0].fit_s4 
    << ", fit_s5 = " << param_h_[0].fit_s5 << std::endl;
}

void RFGap::PrintFromDevice() const
{
  RFGapParameter* tmp = new RFGapParameter;
  CopyDataFromDevice(tmp, param_d_, 1); 
  std::cout << GetName() + ": " << GetType() << ", length = " << GetLength() 
    << ", aper " << GetAperture() << ", freq = " << tmp[0].frequency << ", \n"
    << "\t cell_len_bl = " << tmp[0].cell_length_over_beta_lambda 
    << ", amp = " << tmp[0].amplitude 
    << ", ref_phi = " << tmp[0].phase_ref << ", \n"
    << "\t beam_phi_shift = " << tmp[0].phase_shift
    << ", energy_out = " << tmp[0].energy_out 
    << ", beta_center = " << tmp[0].beta_center << ", \n"
    << "\t dg = " << tmp[0].dg << ", t = " << tmp[0].t 
    << ", tp = " << tmp[0].tp << ", sp = " << tmp[0].sp << ", \n"
    << "\t fit_betac = " << tmp[0].fit_beta_center 
    << ", fit_betamin = " << tmp[0].fit_beta_min 
    << ", fit_t0 = " << tmp[0].fit_t0 << ", \n"
    << "\t fit_t1 = " << tmp[0].fit_t1
    << ", fit_t2 = " << tmp[0].fit_t2
    << ", fit_t3 = " << tmp[0].fit_t3 << ", \n"
    << "\t fit_t4 = " << tmp[0].fit_t4
    << ", fit_t5 = " << tmp[0].fit_t5
    << ", fit_s1 = " << tmp[0].fit_s1 << ", \n"
    << "\t fit_s2 = " << tmp[0].fit_s2
    << ", fit_s3 = " << tmp[0].fit_s3
    << ", fit_s4 = " << tmp[0].fit_s4 
    << ", fit_s5 = " << tmp[0].fit_s5 << std::endl;
  delete tmp;
}

void RFGap::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}
