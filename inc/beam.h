#ifndef BEAM_H
#define BEAM_H
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "py_wrapper.h"

typedef unsigned int uint;

class Beam : public PyWrapper
{
public:
  Beam();
  void AllocateBeam(uint r_num, double r_mass, double r_charge, double r_current);
  void InitPhiAvgGood();
  void InitBeamFromDistribution(std::vector<double>& r_x, 
          std::vector<double>& r_xp, std::vector<double>& r_y, 
          std::vector<double>& r_yp, std::vector<double>& r_phi,
          std::vector<double>& r_w, std::vector<uint>* r_loss = NULL,
          std::vector<uint>* r_lloss = NULL);

  void InitBeamFromFile(std::string r_file);
  void InitBeamFromFile(std::string r_file, double r_mass, double r_charge, 
                        double r_current);
  void InitWaterbagBeam(double r_ax, double r_bx, double r_ex,
    double r_ay, double r_by, double r_ey, double r_az, double r_bz, 
    double r_ez, double r_sync_phi, double r_sync_w, double r_freq, 
    uint r_seed = 0);
  void InitDCBeam(double r_ax, double r_bx, double r_ex, 
                  double r_ay, double r_by, double r_ey, double r_dphi,
                  double r_sync_phi, double r_sync_w, uint r_seed = 0);
  void SaveInitialBeam();
  void SaveIntermediateBeam();
  void RestoreInitialBeam();
  void RestoreIntermediateBeam();
  void FreeBeam();
  void UpdateBeamFromFile(std::string r_file);
  void ConfigureThreads(uint r_blck_size);
  void InitPartials();
  void UpdateLoss();
  void UpdateLongitudinalLoss();
  void UpdateGoodParticleCount();
  void UpdateAvgXp();
  void UpdateAvgYp();
  void UpdateAvgX(bool r_good_only = false);
  void UpdateAvgY(bool r_good_only = false);
  void UpdateAvgPhi(bool r_good_only = false);
  void UpdateAvgRelativePhi(bool r_good_only = false);
  void UpdateAvgW(bool r_good_only = false);
  void UpdateSigXp();
  void UpdateSigYp();
  void UpdateSigX(bool r_good_only = false);
  void UpdateSigY(bool r_good_only = false);
  void UpdateSigPhi();
  void UpdateSigRelativePhi(bool r_good_only = false);
  void UpdateSigW(bool r_good_only = false);
//  void UpdateSigR();
  void UpdateEmittance();
  // used in spch, update avgs of x, y and phi, sig of x and y
  void UpdateStatForSpaceCharge();
  void UpdateStatForPlotting();
  void UpdateAvgXY();
  void UpdateMaxR2Phi();
  void UpdateRelativePhi(bool r_use_good = false);
  void Print(std::string r_file, std::string r_msg = ""); 
  void Print(uint r_indx); 
  void PrintSimple(); 
  void SetRefEnergy(double);
  void SetRefPhase(double);
  double GetRefEnergy() const;
  double GetRefPhase() const;

  double GetAvgX(bool r_good_only = false) const;
  double GetAvgY(bool r_good_only = false) const;
  double GetAvgPhi(bool r_good_only = false) const;
  double GetAvgRelativePhi() const;
  double GetAvgW(bool r_good_only = false) const;

  double GetSigX(bool r_good_only = false) const;
  double GetSigY(bool r_good_only = false) const;
  double GetSigPhi() const;
  double GetSigRelativePhi(bool r_good_only = false) const;
  double GetSigW(bool r_good_only = false) const;
//  double GetSigR() const;

  double GetEmittanceX() const;
  double GetEmittanceY() const;
  double GetEmittanceZ() const;

  double GetMaxPhi() const;

  uint GetLossNum() const;
  uint GetLongitudinalLossNum() const;
  uint GetGoodParticleNum() const;

  std::vector<double> GetX() const;
  std::vector<double> GetXp() const;
  std::vector<double> GetY() const;
  std::vector<double> GetYp() const;
  std::vector<double> GetPhi() const;
  std::vector<double> GetRelativePhi() const;
  std::vector<double> GetW() const;
  std::vector<uint> GetLoss() const;
  std::vector<uint> GetLongitudinalLoss() const;
  
  void ApplyCut(char, double, double);

  void ShiftX(double r_val);
  void ShiftXp(double r_val);
  void ShiftY(double r_val);
  void ShiftYp(double r_val);
  void ShiftPhi(double r_val);
  void ShiftW(double r_val);
  void ChangeFrequency(double r_freq);

// device pointers to arrays, size = num_particle
  double* x;
  double* y;
  double* phi;
  double* phi_r; // phase relative to reference particle
  double* xp;
  double* yp;
  double* w;
  uint* loss;   // transverse loss
  uint* lloss;  // longitudinal loss
  // device pointer to a data
  double* x_avg;
  double* x_avg_good;
  double* xp_avg;
  double* y_avg;
  double* y_avg_good;
  double* yp_avg;
  double* phi_avg;
  double* phi_avg_r;
  double* phi_avg_good;
  double* w_avg;
  double* w_avg_good;
  double* x_sig;
  double* x_sig_good;
  double* xp_sig;
  double* y_sig;
  double* y_sig_good;
  double* yp_sig;
  double* phi_sig;
  double* phi_sig_good;
  double* phi_sig_r; // for relative phase
  double* w_sig;
  double* w_sig_good;
  double* x_emit;
  double* x_emit_good;
  double* y_emit;
  double* y_emit_good;
  double* z_emit;
  double* z_emit_good;
  double* r_max;
//  double* r_sig;
  double* abs_phi_max;
  double* ref_energy;
  double* ref_phase;
  uint* num_loss;
  uint* num_lloss;
  uint* num_good;
  uint* partial_int1;
  double* partial_double2; // size = (grid_size + 1)*2
  double* partial_double3; // size = (grid_size + 1)*3
  // host data
  double mass;
  double charge;
  double current;
  double freq;
  uint num_particle;
  uint grid_size;
  uint blck_size;
  double design_w;
private:
  // copy of the initial beam, x, xp, y, yp, phi, w
  std::vector<double*> beam_0;
  uint* loss_0;
  uint* lloss_0;
  double design_w_0;
  double freq_0;
  // copy of the intermediate beam
  std::vector<double*> beam_1;
  uint* loss_1;
  uint* lloss_1;
  double design_w_1;
  double freq_1;
  void SaveBeam(std::vector<double*>&, uint*&, uint*&);
  void RestoreBeam(std::vector<double*>&, uint*&, uint*&);
};

#endif
