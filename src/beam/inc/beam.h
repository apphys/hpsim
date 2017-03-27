#ifndef BEAM_H
#define BEAM_H
#include <string>
#include <vector>
#include "py_wrapper.h"

typedef unsigned int uint;

/*!
 * \class Beam beam.h
 * \breif Beam class holds all the information about beam.
 *
 * Beam coordinates reside on the device. <br>
 * Eight coordinates: x, xp, y, yp, phase, kinetic_energy, 
 * transverse_loss, longitudinal_loss. <br>
 * For transverse_loss, nonzero numbers denote the indices of the beam line
 * element (starting from 0)  at which the particles become lost. 
 * Zero means the particle is not lost transversely.
 * Similarly for longitudinal_loss, zero means the partilce is not
 * lost longitudinally, but if a particle is lost longitudinally,
 * this field should be 1. Currently, no beam line element index
 * is recorded in this field. This feature can be added in the 
 * future. The longitudinal_loss is used to screen beam for 
 * space charge calculations. Because the space charge needs to be
 * calculated in the beam frame, including the longitudinally lost 
 * beam can make the calculation of the beam center speed less 
 * accurate. ith the longitudinal_loss field, one can exclude
 * the longitudinally lost particles from the space charge calculation. 
 *
 * All the parallel reduction kernels of the beam class uses the algorithm
 * from <br>
 * Mark Harris, "Optimizing Parallel Reduction in CUDA", NIVDIA, 2007.
 */
class Beam : public PyWrapper
{
public:
  Beam();
  Beam(std::string r_file);
  Beam(uint r_num, double r_mass, double r_charge, double r_current);
  ~Beam();
  void AllocateBeam(uint r_num, double r_mass, double r_charge, 
		    double r_current);
  void InitBeamFromFile(std::string r_file);
  void InitBeamFromDistribution(std::vector<double>& r_x, 
      std::vector<double>& r_xp, std::vector<double>& r_y, 
      std::vector<double>& r_yp, std::vector<double>& r_phi,
      std::vector<double>& r_w, std::vector<uint>* r_loss = NULL,
      std::vector<uint>* r_lloss = NULL);
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
  void UpdateEmittance();
  void UpdateStatForSpaceCharge();
  void UpdateStatForPlotting();
  void UpdateAvgXY();
  void UpdateMaxRPhi();
  void UpdateRelativePhi(bool r_use_good = false);
  void PrintToFile(std::string r_file, std::string r_msg = ""); 
  void Print(uint r_indx); 
  void PrintSimple(); 
  void SetNumThreadsPerBlock(uint r_blck_size);
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
  //double GetSigR() const;
  double GetEmittanceX() const;
  double GetEmittanceY() const;
  double GetEmittanceZ() const;
  double GetMaxPhi() const;
  uint GetLossNum() const;
  uint GetLongitudinalLossNum() const;
  uint GetGoodParticleNum() const;
  std::vector<double> GetX() const;
  void GetX(double* r_out);
  std::vector<double> GetXp() const;
  void GetXp(double* r_out);
  std::vector<double> GetY() const;
  void GetY(double* r_out);
  std::vector<double> GetYp() const;
  void GetYp(double* r_out);
  std::vector<double> GetPhi() const;
  void GetPhi(double* r_out);
  std::vector<double> GetRelativePhi() const;
  void GetRelativePhi(double* r_out);
  std::vector<double> GetW() const;
  void GetW(double* r_out);
  std::vector<uint> GetLoss() const;
  void GetLoss(uint* r_out);
  std::vector<uint> GetLongitudinalLoss() const;
  void GetLongitudinalLoss(uint* r_out);
  
  void ApplyCut(char, double, double);

  void ShiftX(double r_val);
  void ShiftXp(double r_val);
  void ShiftY(double r_val);
  void ShiftYp(double r_val);
  void ShiftPhi(double r_val);
  void ShiftW(double r_val);
  void ChangeFrequency(double r_freq);

  // host data
  //! Number of macro-particles, on host 
  uint num_particle; 
  //! Particle rest mass, on host
  double mass;  
  //! Particle charge, on host
  double charge; 
  //! Beam current, on host in [A]
  double current; 
  //! Frequency, used to define phase, on host
  double freq; 
  //! Design energy, update during simulation, on host
  double design_w; 
  //! Number of threads per block used in kernel launching
  uint blck_size; 
  //! Number of blocks used in kernel launching
  uint grid_size; 
  //! x coordinates, device pointer to an array of size num_particle
  double* x; 
  //! y coordinates, devcie pointer to an array of size num_particle
  double* y; 
  //! xp coordinates, device pointer to an array of size num_particle
  double* xp; 
  //! yp coordinates, device pointer to an array of size num_particle
  double* yp; 
  //! Absolute phases, device pointer to an array of size num_particle
  double* phi; 
  //! Phase relative to reference particle, device pointer to an array 
  //! of size num_particle 
  double* phi_r; 
  //! Kinetic energies, device pointer to an array of size num_particle
  double* w; 
  //! Transverse loss, device poiner to an array of size num_particle
  uint* loss; 
  //! Longitudinal loss, device pointer to an array of size num_particle
  uint* lloss; 
  //! x mean, excluding particles transversely lost, device pointer to a number
  double* x_avg; 
  //! x mean, excluding particles either transversely or longitudinally lost, 
  //! device pointer to a number.
  double* x_avg_good; 
  //! xp mean, excluding particles transversely lost, device pointer to a number
  double* xp_avg; 
  //! y mean, excluding particles transversely lost, device pointer to a number
  double* y_avg; 
  //! y mean, excluding particles either transversely or longitudinally lost, 
  //! device pointer to a number
  double* y_avg_good;  
  //! yp mean, excluding particles transversely lost, device pointer to a number
  double* yp_avg; 
  //! Absolute phase mean, excluding particles transversely lost, 
  //! device pointer to a number
  double* phi_avg; 
  //! Relative phase mean, excluding particles transversely lost, 
  //! device pointer to a number
  double* phi_avg_r; 
  //! Absolute phase mean, excluding particles either transversely or 
  //! longitudinally lost, device pointer to a number
  double* phi_avg_good; 
  //! Kinetic energy mean, excluding particles transversely lost, 
  //! device pointer to a number
  double* w_avg; 
  //! Kinetic energy mean, excluding particles either transversely or 
  //! longitudinally lost, device pointer to a number
  double* w_avg_good; 
  //! x std, excluding particles transversely lost, device pointer to a number
  double* x_sig; 
  //! x std, excluding particles either transversely or longitudinally lost, 
  //! device pointer to a number
  double* x_sig_good; 
  //! xp std, excluding particles transversely lost, device pointer to a number
  double* xp_sig; 
  //! y std, excluding particles transversely lost, device pointer to a number
  double* y_sig; 
  //! y std, excluding particles either transversely or longitudinally lost, 
  //! device pointer to a number
  double* y_sig_good; 
  //! yp std, excluding particles transversely lost, device pointer to a number
  double* yp_sig; 
  //! Absolute phase std, excluding particles transversely lost, 
  //! device pointer to a number
  double* phi_sig; 
  //! Absolute phase std, excluding particles either transversely or 
  //! longitudinally lost, device pointer to a number
  double* phi_sig_good; 
  //! Relative phase std, excluding particles transversely lost, 
  //! device pointer to a number
  double* phi_sig_r; 
  //! Kinetic energy std, excluding particles transversely lost, 
  //! device pointer to a number
  double* w_sig; 
  //! Kinetic energy std, excluding particles either transversely or 
  //! longitudinally lost, device pointer to a number
  double* w_sig_good; 
  //! x emittance, excluding particles transversely lost, 
  //! device pointer to a number
  double* x_emit; 
  //! x emittance, excluding particles either transversely or longitudinally 
  //! lost, device pointer to a number
  double* x_emit_good; 
  //! y emittance, excluding particles transversely lost, device pointer 
  //! to a number
  double* y_emit; 
  //! y emittance, excluding particles either transversely or longitudinally 
  //! lost, device pointer to a number
  double* y_emit_good; 
  //! z emittance, excluding particles transversely lost, device 
  //! pointer to a number
  double* z_emit; 
  //! z emittance, excluding particles either transversely or longitudinally 
  //! lost, device pointer to a number
  double* z_emit_good; 
  //! Maximum transverse beam size, r = sqrt(x^2+y^2). excluding particles 
  //! transversely lost, device pointer to a number
  double* r_max; 
  //! Maximum absolute phase, r = sqrt(x^2+y^2). excluding particles 
  //! transversely lost, device pointer to a number
  double* abs_phi_max; 
  //! Number of transversely lost particles, device pointer to a number
  uint* num_loss; 
  //! Number of longitudinally lost particles, device pointer to a number
  uint* num_lloss; 
  //! Number of good particles, which means they are not lost either 
  //! transversely nor longitudinally
  uint* num_good; 
  //! Temporary array used in reduce kernels. size = grid_size + 1
  uint* partial_int1; 
  //! Temporary array used in reduce kernels. size = (grid_size + 1) * 2
  double* partial_double2; 
  //! Temporary array used in reduce kernels. size = (grid_size + 1) * 3
  double* partial_double3; 

private:
  void UpdateBeamFromFile(std::string r_file);
  void InitPhiAvgGood();
  void SaveBeam(std::vector<double*>& r_beam, uint*& r_loss, uint*& r_lloss);
  void RestoreBeam(std::vector<double*>& r_beam, uint*& r_loss, uint*& r_lloss);
  //! Flag to indicate whether the memory is allocated. 
  //! Default to be false. Set to be true only by AllocateBeam() 
  bool init_done; 
  //! Copy of the initial beam, x, xp, y, yp, phi, w coordinates, 
  //! used in save and restore beam to initial condition
  std::vector<double*> beam_0;
  //! Copy of the inital beam transverse loss coordinate, 
  //! used in save and restore beam to initial condition
  uint* loss_0; 
  //! Copy of the inital beam longitudinal loss coordinate, 
  //! used in save and restore beam to initial condition
  uint* lloss_0; 
  //! Copy of the design energy, used in save and restore beam to initial 
  //! condition
  double design_w_0; 
  //! Copy of the frequency, used in save and restore beam to initial condition
  double freq_0; 
  //! Copy of the current, used in save and restore beam to initial condition
  double current_0; 
  //! Copy of the intermediate beam, x, xp, y, yp, phi, w coordinates, used in 
  //! save and restore beam to intermediate condition
  std::vector<double*> beam_1; 
  //! Copy of the intermediate beam transverse loss coordinate, 
  //! used in save and restore beam to intermediate condition
  uint* loss_1; 
  //! Copy of the intermediate beam longitudinal loss coordinate, 
  //! used in save and restore beam to intermediate condition
  uint* lloss_1; 
  //! Copy of the design energy, used in save and restore beam to 
  //! intermediate condition
  double design_w_1; 
  //! Copy of the frequency, used in save and restore beam to 
  //! intermediate condition
  double freq_1; 
  //! Copy of the current, used in save and restore beam to initial condition.
  double current_1; 
};

#endif
