#ifndef BEAMLINE_ELEMENT_H
#define BEAMLINE_ELEMENT_H

#include <string>
#include "dipole_parameter.h"
#include "rfgap_parameter.h"

class Visitor;
class BeamLineElement 
{
public:
  BeamLineElement(std::string, std::string, double r_length = 0.0);
  virtual ~BeamLineElement(){}
  std::string GetName() const;
  std::string GetType() const;
  double GetLength() const;
  double GetAperture() const;
  bool IsMonitorOn() const;
  void SetType(std::string);
  void SetLength(double);
  void SetAperture(double);
  void SetMonitorOn();
  void SetMonitorOff();
  virtual void Print() const = 0;
  virtual void Accept(Visitor*) = 0;
private:
  std::string type_;
  std::string name_;
  double length_;
  double aperture_;
  bool monitor_on_;
};

class Buncher: public BeamLineElement
{
public:
  Buncher(std::string);
  bool IsOn() const;
  double GetVoltage() const;
  double GetFrequency() const;
  double GetPhase() const;
  void TurnOn();
  void TurnOff();
  void SetVoltage(double);
  void SetFrequency(double); 
  void SetPhase(double);
  void Print() const;
  void Accept(Visitor*); 
private:
  bool ison_;
  double voltage_;
  double frequency_;
  double phase_;
};

class Dipole: public BeamLineElement
{
public:
  Dipole(std::string);
  ~Dipole();
  void SetRadius(double);
  void SetAngle(double);
  void SetEdgeAngleIn(double);
  void SetEdgeAngleOut(double);
  void SetHalfGap(double);
  void SetK1(double);
  void SetK2(double);
  void SetFieldIndex(double);
  void SetKineticEnergy(double);
  void Print() const;
  void PrintFromDevice() const;
  void Accept(Visitor*);
private:
  DipoleParameter* param_h_;
  DipoleParameter* param_d_;
};

class Drift: public BeamLineElement
{
public:
  Drift(std::string);
  void Print() const;
  void Accept(Visitor*);
};

class Quad : public BeamLineElement
{
public:
  Quad(std::string);
  double GetGradient() const;
  void SetGradient(double); 
  void Print() const;
  void Accept(Visitor*);
private:
  double gradient_;
};

class RFGap: public BeamLineElement
{
public:
  RFGap(std::string, std::string); // type = "RFGap-DTL" or "RFGap-CCL"
  ~RFGap();
  RFGapParameter* GetParameters() const;
  RFGapParameter* GetParametersOnDevice() const;
  double GetFrequency() const;
  double GetRefPhase() const;
  double GetEnergyOut() const;
  double GetRFAmplitude() const;
  double GetPhaseShift() const;
  void SetFrequency(double);
  void SetCellLengthOverBetaLambda(double);
  void SetRFAmplitude(double);
  void SetRefPhase(double);
  void SetPhaseShift(double);
  void SetEnergyOut(double);
  void SetBetaCenter(double);
  void SetDg(double); // electric/geometric center shift
  void SetT(double);
  void SetTp(double);
  void SetSp(double);
  // For higherer order transit time factor
  void SetFitBetaCenter(double); 
  void SetFitBetaMin(double); 
  void SetFitT0(double);
  void SetFitT1(double);
  void SetFitT2(double);
  void SetFitT3(double);
  void SetFitT4(double);
  void SetFitT5(double);
  void SetFitS1(double);
  void SetFitS2(double);
  void SetFitS3(double);
  void SetFitS4(double);
  void SetFitS5(double);
  
  void Print() const;
  void PrintFromDevice() const;
  void Accept(Visitor*);
private:
  RFGapParameter* param_h_;
  RFGapParameter* param_d_;
};

class Rotation: public BeamLineElement
{
public:
  Rotation(std::string);
  double GetAngle() const;
  void SetAngle(double);
  void Print() const;
  void Accept(Visitor*);
private:
  double angle_;
};

inline  
std::string BeamLineElement::GetName() const
{
  return name_;
}
inline
std::string BeamLineElement::GetType() const
{
  return type_;
}
inline
double BeamLineElement::GetLength() const
{
  return length_;
}
inline
double BeamLineElement::GetAperture() const
{
  return aperture_;
}
inline 
bool BeamLineElement::IsMonitorOn() const
{
  return monitor_on_;
}
inline
void BeamLineElement::SetType(std::string r_type)
{
  type_ = r_type;
}
inline 
void BeamLineElement::SetLength(double r_length)
{
  length_ = r_length;
}
inline
void BeamLineElement::SetAperture(double r_aper)
{
  aperture_ = r_aper;
}
inline
void BeamLineElement::SetMonitorOn()
{
  monitor_on_ = true;
}
inline
void BeamLineElement::SetMonitorOff()
{
  monitor_on_ = false;
}
inline
bool Buncher::IsOn() const
{
  return ison_;
}
inline
double Buncher::GetVoltage() const
{
  return voltage_;
}
inline
double Buncher::GetFrequency() const
{
  return frequency_;
}
inline
double Buncher::GetPhase() const
{
  return phase_;
}
inline
void Buncher::TurnOn()
{
  ison_ = true;
}
inline
void Buncher::TurnOff()
{
  ison_ = false;
}
inline
void Buncher::SetVoltage(double r_vol)
{
  voltage_ = r_vol;
}
inline
void Buncher::SetFrequency(double r_freq)
{
  frequency_ = r_freq;
}
inline
void Buncher::SetPhase(double r_ph)
{
  phase_ = r_ph;
}
inline
void Dipole::SetRadius(double r_radius)
{
  param_h_->radius = r_radius;
}
inline
void Dipole::SetAngle(double r_angle)
{
  param_h_->angle = r_angle;
}
inline
void Dipole::SetEdgeAngleIn(double r_eangle_in)
{
  param_h_->edge_angle_in = r_eangle_in;
}
inline
void Dipole::SetEdgeAngleOut(double r_eangle_out)
{
  param_h_->edge_angle_out = r_eangle_out;
}
inline
void Dipole::SetHalfGap(double r_half_gap)
{
  param_h_->half_gap = r_half_gap;
}
inline
void Dipole::SetK1(double r_k1)
{
  param_h_->k1 = r_k1;
}
inline
void Dipole::SetK2(double r_k2)
{
  param_h_->k2 = r_k2;
}
inline
void Dipole::SetFieldIndex(double r_findx)
{
  param_h_->field_index = r_findx;
}
inline
void Dipole::SetKineticEnergy(double r_kenergy)
{
  param_h_->kinetic_energy = r_kenergy;
}
inline
double Quad::GetGradient() const
{
  return gradient_;
}
inline
void Quad::SetGradient(double r_grad) 
{
  gradient_ = r_grad;
}
inline
RFGapParameter* RFGap::GetParameters() const
{
  return param_h_;
}
inline
RFGapParameter* RFGap::GetParametersOnDevice() const
{
  return param_d_;
}
inline
double RFGap::GetFrequency() const
{
  return param_h_->frequency;
}
inline
double RFGap::GetRefPhase() const
{
  return param_h_->phase_ref;
}
inline
double RFGap::GetEnergyOut() const
{
  return param_h_->energy_out;
}
inline
double RFGap::GetRFAmplitude() const
{
  return param_h_->amplitude;
}
inline 
double RFGap::GetPhaseShift() const
{
  return param_h_->phase_shift;
}
inline
void RFGap::SetFrequency(double r_freq)
{
  param_h_->frequency = r_freq;
}
inline
void RFGap::SetCellLengthOverBetaLambda(double r_cell_len_bl)
{
  param_h_->cell_length_over_beta_lambda = r_cell_len_bl;
}
inline
void RFGap::SetRFAmplitude(double r_amp)
{
  param_h_->amplitude = r_amp; 
}
inline
void RFGap::SetRefPhase(double r_phase_ref)
{
  param_h_->phase_ref = r_phase_ref;
}       
inline
void RFGap::SetPhaseShift(double r_phase_shift)
{
  param_h_->phase_shift = r_phase_shift;
}
inline
void RFGap::SetEnergyOut(double r_energy_out)
{
  param_h_->energy_out = r_energy_out;
}
inline
void RFGap::SetBetaCenter(double r_beta_center)
{
  param_h_->beta_center = r_beta_center;
}
inline
void RFGap::SetDg(double r_dg)
{
   param_h_->dg = r_dg;
}
inline
void RFGap::SetT(double r_t)
{
  param_h_->t = r_t;
}
inline
void RFGap::SetTp(double r_tp)
{
   param_h_->tp = r_tp;
}
inline
void RFGap::SetSp(double r_sp)
{
  param_h_->sp = r_sp;
}
inline
void RFGap::SetFitBetaCenter(double r_fbetac)
{
  param_h_->fit_beta_center = r_fbetac;
}
inline
void RFGap::SetFitBetaMin(double r_fbeta_min)
{
  param_h_->fit_beta_min = r_fbeta_min;
}
inline
void RFGap::SetFitT0(double r_fitt0)
{
  param_h_->fit_t0 = r_fitt0;
}
inline 
void RFGap::SetFitT1(double r_fitt1)
{
  param_h_->fit_t1 = r_fitt1;
}
inline
void RFGap::SetFitT2(double r_fitt2)
{
  param_h_->fit_t2 = r_fitt2;
}
inline
void RFGap::SetFitT3(double r_fitt3)
{
  param_h_->fit_t3 = r_fitt3;
}
inline
void RFGap::SetFitT4(double r_fitt4)
{
  param_h_->fit_t4 = r_fitt4;
}
inline
void RFGap::SetFitT5(double r_fitt5)
{
  param_h_->fit_t5 = r_fitt5;
}
inline
void RFGap::SetFitS1(double r_fits1)
{
  param_h_->fit_s1 = r_fits1;
}
inline
void RFGap::SetFitS2(double r_fits2)
{
  param_h_->fit_s2 = r_fits2;
}
inline
void RFGap::SetFitS3(double r_fits3)
{
  param_h_->fit_s3 = r_fits3;
}
inline
void RFGap::SetFitS4(double r_fits4)
{
  param_h_->fit_s4 = r_fits4;
}
inline
void RFGap::SetFitS5(double r_fits5)
{
  param_h_->fit_s5 = r_fits5;
}
inline
double Rotation::GetAngle() const
{
  return angle_;
}
inline 
void Rotation::SetAngle(double r_angle) 
{
  angle_ = r_angle;
}
#endif
