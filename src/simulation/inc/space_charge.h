#ifndef SPACE_CHARGE_H
#define SPACE_CHARGE_H

#include <cuda_runtime_api.h> // for double2
#include "beam.h"
#include "space_charge_parameter.h"
#include "py_wrapper.h"

/*!
 * \brief Space charge base class.
 */
class SpaceCharge : public PyWrapper
{
public:
  SpaceCharge(uint r_nx, uint r_nz, uint r_ny = 0);
  virtual ~SpaceCharge(){}
  virtual void Start(Beam*, double r_length) = 0;
  virtual void SetMeshSize(uint r_nx, uint r_nz, uint r_ny = 0)
  {
    nx_ = r_nx; ny_ = r_ny; nz_ = r_nz;
  }
  bool Is2D() const
  {
    if (ny_ == 0.0) return true;
    return false;
  }
  uint GetNx() const
  {
    return nx_;
  }
  uint GetNy() const
  {
    return ny_;
  }
  uint GetNz() const
  {
    return nz_;
  }
  double GetFraction() const
  {
    return fraction_;
  }
  void SetFraction(double r_frac)
  {
    fraction_ = r_frac;
  }
  uint GetAdjBunch() const
  {
    return adj_bunch_;
  }
  void SetAdjBunch(uint r_adjb)
  {
    adj_bunch_ = r_adjb;
  }
  double GetInterval() const
  {
    return interval_;
  }
  void SetInterval(double r_interval)
  {
    interval_ = r_interval;
    interval_init_ = r_interval;
  }
  double GetAdjBunchCutoffW() const
  {
    return adj_bunch_w_;
  }
  void SetAdjBunchCutoffW(double r_w)
  {
    adj_bunch_w_ = r_w;
  }
  double GetMeshSizeCutoffW() const
  {
    return mesh_w_;
  }
  void SetMeshSizeCutoffW(double r_w)
  {
    mesh_w_ = r_w;
  }
  double GetRemeshThreshold() const
  {
    return remesh_val_;
  }
  void SetRemeshThreshold(double r_val)
  {
    remesh_val_ = r_val;
  }
protected:
  //! Horizontal mesh size, transverse mesh size for SCHEFF
  uint nx_;
  //! Vertical mesh size
  uint ny_;
  //! longitudinal mesh size
  uint nz_;
  //! Initial mesh sizes
  uint nx_init_, ny_init_, nz_init_;
  //! Number of adjacent bunches
  uint adj_bunch_;
  //! Fraction of effective current due to space charge compensation
  double fraction_;
  //! Maximum spacing between space charge kicks
  double interval_;
  //! Initial maximum spacing between space charge kicks
  double interval_init_;
  //! Cutoff energy (MeV) above which the adjacent bunches are 
  //! no longer used in space charge calculation
  double adj_bunch_w_;
  //! Cutoff energy for the beam at which the mesh size will
  //! decrease by nr/2 and nz/2 and interval increase by 4.This enables 
  //! automatic transition to faster space charge calculation
  double mesh_w_;
  //! Previous transverse sigma 
  double sigr_old_;
  //! Previous longitudinal sigma 
  double sigz_old_;
  //! Previous gamma
  double gm_old_;
  //! Previous number of good particles (not lost tranversely or longitudinally)
  uint good_cnt_old_;
  //! Remeshing factor (concept borrowed from ASTRA 
  //! (http://www.desy.de/~mpyflo/)
  //! default is 0.05) where zero means remesh before every space-charge kick,
  //! positive means adaptive algorithm determines how much beam shape can 
  //! change before mesh must be redone
  double remesh_val_;
};

/*!
 * \brief 2D SCHEFF (Space CHarge EFFect) class. Calculates the radial and 
 * longitudinal space charge forces for cylindrically symmetric beam 
 * distribution.
 *
 * Algorithm details see 
 * X. Pang, L. Rybarcyk, "GPU Accelerated Online Beam Dynamics Simulator 
 * for Linear Particle Accelerators", Computer Physics Communications, 
 * 185, pp. 744-753, 2014.
 */
class Scheff : public SpaceCharge
{
public:
  Scheff(uint, uint, int r_adj_bunch = 0);
  ~Scheff();
  void Start(Beam*, double r_length);
  virtual void SetMeshSize(uint r_nx, uint r_nz, uint r_ny = 0);

private:
#ifdef DOUBLE_PRECISION
  //! Device pointer to charge density table (2D, nz by nx) 
  //! consecutive threads share the same nz 
  double* d_bin_tbl_;
  //! Device pointer to final Green's function table(2D, nz by nx) 
  //! consecutive threads share the same nz 
  double2* d_fld_tbl2_; 
#else
  //! Device pointer to charge density table (2D, nz by nx), 
  //! consecutive threads share the same nz 
  float* d_bin_tbl_;   
  //! Device pointer to final Green's function table (2D, nz by nx) 
  //! consecutive threads share the same nz 
  float2* d_fld_tbl2_;
#endif
  //! Device pointer to the intermediate Green's function table 
  //! (3D, nx+1 by nx by nz) consecutive threads share the same nx
  double2* d_fld_tbl1_; 
};

#endif
