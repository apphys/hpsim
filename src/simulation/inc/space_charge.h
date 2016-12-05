#ifndef SPACE_CHARGE_H
#define SPACE_CHARGE_H
#include <cuda_runtime_api.h> // double2
#include "beam.h"
#include "space_charge_parameter.h"
#include "py_wrapper.h"

class SpaceCharge : public PyWrapper
{
public:
  SpaceCharge(uint r_nx, uint r_nz, uint r_ny = 0) 
    : PyWrapper(), nx_(r_nx), ny_(r_ny), nz_(r_nz), 
      adj_bunch_(0), fraction_(1.0), interval_(0.01),
      adj_bunch_w_(0.0), sigr_old_(0.0), sigz_old_(0.0),
      gm_old_(0.0), good_cnt_old_(0), dist_accum_(0.0),
      mesh_w_(0.0), nx_init_(r_nx), ny_init_(r_ny), 
      nz_init_(r_nz), interval_init_(0.01), remesh_val_(0.05)
  {}
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
  uint nx_, ny_, nz_;
  uint nx_init_, ny_init_, nz_init_;
  uint adj_bunch_;
  double fraction_;
  double interval_;
  double interval_init_;
  double adj_bunch_w_;
  double mesh_w_;
  double sigr_old_;
  double sigz_old_;
  double gm_old_;
  uint good_cnt_old_;
  double dist_accum_;
  double remesh_val_;
};

class Scheff : public SpaceCharge
{
public:
  Scheff(uint, uint, int r_adj_bunch = 0);
  ~Scheff();
  void Start(Beam*, double r_length);
  virtual void SetMeshSize(uint r_nx, uint r_nz, uint r_ny = 0);

private:
  // for dim, the first num is what the consecutive elems share 
#ifdef DOUBLE_PRECISION
  double* d_bin_tbl_;
  double2* d_fld_tbl2_; // nz*nr
#else
  float* d_bin_tbl_;    // nz*nr 
  float2* d_fld_tbl2_; // nz*nr
#endif
  double2* d_fld_tbl1_; // (nr+1)*nr*nz
};

#endif
