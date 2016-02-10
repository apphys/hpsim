#include <iostream>
#include "beamline_element.h"
#include "pin_mem.h"
#include "visitor.h"

BeamLineElement::BeamLineElement(std::string r_name, std::string r_type, 
  double r_length) : name_(r_name), type_(r_type), length_(r_length), 
  aperture_(0.0), monitor_on_(false)
{
}

ApertureCircular::ApertureCircular(std::string r_name) :
  BeamLineElement(r_name, "ApertureC"), isin_(false)
{
}

void ApertureCircular::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

void ApertureCircular::Print() const
{
  std::cout << GetName() << ": " << GetType() 
    << ", aper = " << GetAperture() 
    << "in/out = " << (IsIn() ? "in" : "out")
    << std::endl;
}

ApertureRectangular::ApertureRectangular(std::string r_name) :
  BeamLineElement(r_name, "ApertureR"), aperture_x_left_(0.0),
  aperture_x_right_(0.0), aperture_y_left_(0.0),
  aperture_y_right_(0.0), isin_(false)
{
}

void ApertureRectangular::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

void ApertureRectangular::Print() const
{
  std::cout << GetName() << ": " << GetType() 
    << ", aper_x_l = " << GetApertureXLeft() 
    << ", aper_x_r = " << GetApertureXRight() 
    << ", aper_y_l = " << GetApertureYLeft() 
    << ", aper_y_r = " << GetApertureYRight() 
    << "in/out = " << (IsIn() ? "in" : "out")
    << std::endl;
}

Buncher::Buncher(std::string r_name) : BeamLineElement(r_name, "Buncher")
{
}

void Buncher::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

void Buncher::Print() const
{
  std::cout << GetName() << ": " << GetType()
    << ", voltage = " << GetVoltage() << ", freq = " << GetFrequency()
    << ", phase = " << GetPhase() << ", on_off = "
    << (IsOn() ? "on" : "off") << std::endl;
}

Diagnostics::Diagnostics(std::string r_name) : 
  BeamLineElement(r_name, "diagnostics")
{
}

void Diagnostics::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

void Diagnostics::Print() const
{
  std::cout << GetName() << ": " << GetType()
    << "monitor_on/off  = " << (IsMonitorOn() ? "on" : "off") << std::endl;
}

Dipole::Dipole(std::string r_name) : BeamLineElement(r_name, "Dipole")
{
  CreateDataOnPinMem(1, &param_h_, &param_d_);
}

Dipole::~Dipole()
{
  if (param_h_ != NULL)
    FreeDataOnPinMem(param_h_); 
}

void Dipole::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

void Dipole::Print() const
{
  std::cout << GetName() << ": " << GetType() << ", length = " << GetLength()
    << ", aper = " << GetAperture() << "\n"
    << "\t radius = " << param_h_->radius << ", angle = " << param_h_->angle
    << ", edge_angle_in = " << param_h_->edge_angle_in << "\n"
    << "\t, edge_angle_out = " << param_h_->edge_angle_out
    << ", half_gap = " << param_h_->half_gap 
    << ", k1 = " << param_h_->k1 << "\n"
    << "\t k2 = " << param_h_->k2 
    << ", field_index = " << param_h_->field_index 
    << ", kenergy = " << param_h_->kinetic_energy
    << std::endl;
}

void Dipole::PrintFromDevice() const
{
  DipoleParameter* tmp = new DipoleParameter;
  CopyDataFromDevice(tmp, param_d_, 1); 
  std::cout << GetName() << ": " << GetType() << ", length = " << GetLength()
    << ", aper = " << GetAperture() << "\n"
    << "\t radius = " << tmp->radius << ", angle = " << tmp->angle
    << ", edge_angle_in = " << tmp->edge_angle_in << "\n"
    << "\t, edge_angle_out = " << tmp->edge_angle_out
    << ", half_gap = " << tmp->half_gap 
    << ", k1 = " << tmp->k1 << "\n"
    << "\t k2 = " << tmp->k2 
    << ", field_index = " << tmp->field_index 
    << ", kenergy = " << tmp->kinetic_energy
    << std::endl;
  delete tmp;
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

SpaceChargeCompensation::SpaceChargeCompensation(std::string r_name) :
  BeamLineElement(r_name, "SpchComp")
{
}

void SpaceChargeCompensation::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

void SpaceChargeCompensation::Print() const
{
  std::cout << GetName() << ": " << GetType() << ", fraction= "
    << GetFraction() << std::endl;
}

Steerer::Steerer(std::string r_name) : BeamLineElement(r_name, "Steerer")
{
}

void Steerer::Accept(Visitor* r_visitor)
{
  r_visitor->Visit(this);
}

void Steerer::Print() const
{
  std::cout << GetName() << ": " << GetType() << ", bl_h = "
    << GetIntegratedFieldHorizontal() 
    << ", bl_v = " << GetIntegratedFieldVertical()
    << std::endl;
}


