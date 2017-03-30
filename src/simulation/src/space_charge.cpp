#include <cmath>
#include <iostream>
#include "space_charge.h"
#include "space_charge_cu.h"
#include "space_charge_parameter.h"

/*!
 * \brief Constructor
 * \param r_nx Horizontal mesh size. Transverse mesh size for SCHEFF.
 * \param r_ny Vertical mesh size.
 * \param r_nz Longitudinal mesh size.
 */
SpaceCharge::SpaceCharge(uint r_nx, uint r_nz, uint r_ny) 
  : PyWrapper(), nx_(r_nx), ny_(r_ny), nz_(r_nz), 
    adj_bunch_(0), fraction_(1.0), interval_(0.01),
    adj_bunch_w_(0.0), sigr_old_(0.0), sigz_old_(0.0),
    gm_old_(0.0), good_cnt_old_(0), mesh_w_(0.0), 
    nx_init_(r_nx), ny_init_(r_ny), nz_init_(r_nz), 
    interval_init_(0.01), remesh_val_(0.05)
{
}

/*!
 * \brief Constructor
 * \param r_nr Transverse mesh size, nx = nr.
 * \param r_nz Longitudinal mesh size.
 * \param r_adj_bunch Number of adjacent bunches. 
 *
 * \callgraph
 */
Scheff::Scheff(uint r_nr, uint r_nz, int r_adj_bunch)
  : SpaceCharge(r_nr, r_nz)
{
  adj_bunch_ = r_adj_bunch;
  SpaceChargeParam param;
  param.nx = r_nr;
  param.nz = r_nz;
  param.ny = 0;
  Initialize(param, &d_fld_tbl1_, &d_fld_tbl2_, &d_bin_tbl_);
}

/*!
 * \brief Destructor. Free all the data on device.
 *
 * \callgraph
 */
Scheff::~Scheff()
{
  FreeMeshTables(d_fld_tbl1_, d_fld_tbl2_, d_bin_tbl_);
  std::cout << "Scheff is freed." << std::endl;
}

/*!
 * \brief Set the mesh dimensions. 
 * \param r_nx Horizontal mesh size. Transverse mesh size for SCHEFF.
 * \param r_ny Vertical mesh size.
 * \param r_nz Longitudinal mesh size.
 *
 * \callgraph
 */
void Scheff::SetMeshSize(uint r_nx, uint r_nz, uint r_ny)
{
  if (r_nx != nx_ || r_nz != nz_)
  {
    FreeMeshTables(d_fld_tbl1_, d_fld_tbl2_, d_bin_tbl_);
    nx_ = r_nx; nz_ = r_nz;
    SpaceChargeParam param;
    param.nx = r_nx;
    param.nz = r_nz;
    param.ny = 0;
    Initialize(param, &d_fld_tbl1_, &d_fld_tbl2_, &d_bin_tbl_);
  }
}

/*!
 * \brief SCHEFF kernel.
 *
 * Thread management is handled by the simulation engine.
 * 
 * \callgraph
 */
void Scheff::Start(Beam* r_beam, double r_length)
{
  r_beam->UpdateStatForSpaceCharge();

  uint good_cnt = r_beam->GetGoodParticleNum();
  double sigphi = r_beam->GetSigRelativePhi(true);
  double sigx = r_beam->GetSigX(true);
  double sigy = r_beam->GetSigY(true);
  double ke = r_beam->GetAvgW(true);
  double gamma = (double)ke / r_beam->mass + 1.0;
  double beta = std::sqrt(1.0 - 1.0 / (gamma * gamma));
  double lmbd = 299.792458 / r_beam->freq;
  double sigz = sigphi * 0.5 / 3.14159265 * beta * lmbd;
  double sigr = std::sqrt(sigx * sigx + sigy * sigy);
  double aratio = sigz / sigr * gamma;
  double cratior = sigr * sigz;
  double cratioz = sigz * sigz * gamma * gamma;
  double cratior_old = sigr_old_ * sigz_old_;
  double cratioz_old = sigz_old_ * sigz_old_ * gm_old_*gm_old_;
  double adj_bunch, nr, nz;
  if(adj_bunch_w_ == 0.0 || ke < adj_bunch_w_)
    adj_bunch = adj_bunch_;
  else
    adj_bunch = 0;
  
  if(mesh_w_ != 0.0)
  {
    if (ke > mesh_w_ && (nx_ != nx_init_ / 2 || nz_ != nz_init_ / 2))
      SetMeshSize(nx_init_ / 2, nz_init_ / 2);
    else if (ke <= mesh_w_ && (nx_ != nx_init_ || nz_ != nz_init_))
      SetMeshSize(nx_init_, nz_init_);
    if (ke > mesh_w_ && interval_ != interval_init_ * 4)
      interval_ = interval_init_ * 4;
    else if (ke <= mesh_w_ && interval_ != interval_init_)
      interval_ = interval_init_;
  }
  
  SetParameters(r_beam, nx_, nz_, fraction_, adj_bunch,  beta);

  if (cratior_old == 0.0 || cratioz_old == 0.0 || 
    std::abs(cratior - cratior_old) / cratior_old > remesh_val_ || 
    std::abs(cratioz - cratioz_old) / cratioz_old > remesh_val_)
  {
    sigr_old_ = sigr;
    sigz_old_ = sigz;
    gm_old_ = gamma;
    good_cnt_old_ = good_cnt;
    ResetBinTbl(d_bin_tbl_);
    ResetFldTbl(d_fld_tbl2_);
    UpdateMeshKernelCall();
    UpdateTblsKernelCall(d_fld_tbl1_, d_bin_tbl_);
    UpdateFinalFldTblKernelCall(d_fld_tbl2_, d_bin_tbl_, d_fld_tbl1_);
    KickBeamKernelCall(d_fld_tbl2_, r_length);
  }
  else
    KickBeamKernelCall(d_fld_tbl2_, r_length, sigr / sigr_old_, 
      sigz / sigz_old_, good_cnt/good_cnt_old_, gamma / gm_old_);
}
