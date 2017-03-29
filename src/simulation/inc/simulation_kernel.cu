#ifndef SIMULATION_KERNEL_CU
#define SIMULATION_KERNEL_CU

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include "simulation_parameter.h"
#include "rfgap_parameter.h"
#include "beam.h"
#include "constant.h"

__constant__ SimulationConstOnDevice d_const;
__device__ double d_wavelen = 0.0;

__device__
double Bessi0c(double r_x)
{
  double ans, Cell_y, ax = abs(r_x);
  if(ax < 3.75)
  {
    Cell_y = r_x / 3.75;
    Cell_y = Cell_y * Cell_y;
    ans = 1.0 + Cell_y * (3.5156229 + Cell_y * (3.0899424 + Cell_y * 
      (1.2067492 + Cell_y * (0.2659732 + Cell_y * (0.360768e-1 + 
      Cell_y * 0.45813e-2)))));
  }
  else
  {
    Cell_y = 3.75 / ax;
    ans = (exp(ax) * rsqrt(ax)) * (0.39894228 + Cell_y * (0.1328592e-1 +
      Cell_y * (0.225319e-2 + Cell_y * (-0.157565e-2 + Cell_y * (0.916281e-2 +
      Cell_y * (-0.2057706e-1 + Cell_y * (0.2635537e-1 + Cell_y *
      (-0.1647633e-1 + Cell_y * 0.392377e-2))))))));
  }
  return ans;
}

__device__
double Bessi1c(double r_x)
{
  double Cell_y, ans, ax = abs(r_x);
  if (ax < 3.75)
  {
    Cell_y = r_x / 3.75;
    Cell_y = Cell_y * Cell_y;
    ans = ax * (0.5 + Cell_y * (0.87890594 + Cell_y * (0.51498869 + Cell_y * 
      (0.15084934 + Cell_y * (0.2658733e-1 + Cell_y * (0.301532e-2 + Cell_y * 
      0.32411e-3))))));
  }
  else
  {
    Cell_y = 3.75 / ax;
    ans = 0.2282967e-1 + Cell_y * (-0.2895312e-1 + Cell_y *
          (0.1787654e-1 - Cell_y * 0.420059e-2));
    ans = 0.39894228 + Cell_y * (-0.3988024e-1 + Cell_y *
          (-0.362018e-2 + Cell_y * (0.163801e-2 + Cell_y *
          (-0.1031555e-1 + Cell_y * ans))));
    ans = ans * (exp(ax) * rsqrt(ax));
  }
  if (r_x < 0.0)
    return -ans;
  else
    return ans;
}

__device__
double Bessi1p(double r_x)
{
  if (r_x != 0.0)
    return Bessi0c(r_x) - Bessi1c(r_x) / r_x;
  else
    return 0.5;
}

__global__
void SimulateTransportKernel(double* r_ref_phase, double* r_ref_energy, 
  double r_freq, double r_energy = 0.0, double r_phi  = 0.0)
{
  d_wavelen = CLIGHT / r_freq;
  if(r_energy != 0.0)
    r_ref_energy[0] = r_energy;
  if(r_phi != 0.0)
    r_ref_phase[0] = r_phi;
}

__global__
void SimulatePartialDriftKernel(double* r_x, double* r_y, double* r_phi, 
  double* r_xp, double* r_yp, double* r_w, uint* r_loss, double r_hf_spch_len, 
  double r_aper, uint r_elem_indx)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  while(index < d_const.num_particle)
  {
    if(r_loss[index] == 0)
    {
      double aper2 = r_aper * r_aper;
      // transport hf spch_len
      double x = r_x[index];
      double y = r_y[index];
      x += r_xp[index] * r_hf_spch_len;
      y += r_yp[index] * r_hf_spch_len;
      // check for loss
      if(aper2 != 0.0 && x * x + y * y > aper2)
        r_loss[index] = r_elem_indx;
      else
      {
        r_x[index] = x;
        r_y[index] = y;
        double gm1 = r_w[index] / d_const.mass;
        double bt = sqrt(gm1 * (gm1 + 2.0))/(gm1 + 1.0);
	// absolute phase
        r_phi[index] += TWOPI * r_hf_spch_len / (d_wavelen * bt); 
      }
    }// if loss
    index += stride;
  }// while
}

__global__
void SimulateSteererKernel(double* r_xp, double* r_yp, 
  double* r_w, uint* r_loss, double r_blh, double r_blv)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  while(index < d_const.num_particle)
  {
    if(r_loss[index] == 0)
    {
      double gm1 = r_w[index] / d_const.mass;  
      double btgm = sqrt(gm1 * (gm1 + 2.0)); 
      double p = btgm * d_const.mass / CLIGHT;
      if (r_blh > 1e-10)
        r_xp[index] += r_blh * d_const.charge / p; 
      if (r_blv > 1e-10)
        r_yp[index] += r_blv * d_const.charge / p; 
    }
    index += stride;
  }
}

__global__ 
void SimulateHalfQuadKernel(double* r_x, double* r_y, double* r_phi, 
  double* r_xp, double* r_yp, double* r_w, uint* r_loss, double r_length, 
  double r_aper, double r_gradient, uint r_elem_indx)
{
  double aper2 = r_aper;
  aper2 *= aper2;
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  double qg = r_gradient * d_const.charge;
  while(index < d_const.num_particle)
  {
    if(r_loss[index] == 0)
    {
      double x = r_x[index];
      double y = r_y[index];
      double xp = r_xp[index];
      double yp = r_yp[index];
      double phi = r_phi[index];
      double w = r_w[index];

      if(qg == 0)
      {
        r_x[index] = x + xp * r_length * 0.5;
        r_y[index] = y + yp * r_length * 0.5;
      }
      else 
      {
        double gamma1 = w / d_const.mass;
        double bg = sqrt(gamma1 * (gamma1 + 2.0) * (1.0 + xp * xp + yp * yp));
        double absqg = qg > 0.0 ? qg : -qg;
        double k = sqrt(absqg / (bg * d_const.mass) *CLIGHT);
        double inv_k = 1.0 / k;
        double kds = r_length * 0.25 * k;
        double s, c, sh = sinh(kds), ch = cosh(kds);
        sincos(kds, &s, &c);
        double  xx, yy, sovk = s * inv_k, shovk = sh * inv_k, sk = s * k, 
		shk = sh * k;
        if(qg> 0.0)
        {
          xx = c * x + sovk * xp; xp = -sk * x + c * xp; x = xx;
          yy = ch * y + shovk * yp; yp = shk * y + ch * yp; y = yy;
        }
        else
        {
          xx = ch * x + shovk * xp; xp = shk * x + ch * xp; x = xx;
          yy = c * y + sovk * yp; yp = -sk * y + c * yp; y=yy;
        } 
        if(qg > 0.0)
        {
          xx = c * x + sovk * xp; xp = -sk * x + c * xp; x = xx;
          yy = ch * y + shovk * yp; yp = shk * y + ch * yp; y = yy;
        }
        else
        {
          xx = ch * x + shovk * xp; xp = shk * x + ch * xp; x = xx;
          yy = c * y + sovk * yp; yp = -sk * y + c * yp; y = yy;
        } 
        r_x[index] = x;
        r_y[index] = y;
        r_xp[index] = xp;
        r_yp[index] = yp;
      }
      // at the center & the end of the quad, check for loss
      if(aper2 != 0 && x * x + y * y > aper2)
        r_loss[index] = r_elem_indx;
      else // if not lost, then check phase 
      {
        double gm1 = w / d_const.mass;
        double bt = sqrt(gm1 * (gm1 + 2.0)) / (gm1 + 1.0);
	// absolute phase
        r_phi[index] += TWOPI * r_length * 0.5 / (d_wavelen * bt); 
      }// if aper2
    } // if loss
    index += stride;
  } // while
}

__global__
void UpdateWaveLengthKernel(double r_freq)
{
  d_wavelen = CLIGHT / r_freq; // in meter
}

__global__
void SimulateRFGapFirstHalfKernel(double* r_x, double* r_y, double* r_phi, 
                            double* r_xp, double* r_yp, double* r_w, 
                            uint* r_loss, double r_design_w_in, 
                            double r_phi_in, RFGapParameter* r_elem, 
                            double r_length, double r_qlen1 = 0.0, 
                            double r_qlen2 = 0.0, bool flag_ccl = true)
{
  RFGapParameter gap = *r_elem;
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  while(index < d_const.num_particle)
  {
    if(r_loss[index] == 0)
    {
      double x = r_x[index];
      double y = r_y[index];
      double xp = r_xp[index];
      double yp = r_yp[index];
      double phi = r_phi[index];
      double w = r_w[index];
      double cell_len = r_length + r_qlen1 * 0.5 + r_qlen2 * 0.5;
      double dd1 = 0.5 * cell_len - gap.dg - 0.5 * r_qlen1; 
      r_x[index] = x + dd1 * xp;
      r_y[index] = y + dd1 * yp;
      double wave_len = CLIGHT / gap.frequency;
      d_wavelen = wave_len;
      if(gap.amplitude == 0.0)
      {
        double gm1 = w / d_const.mass;
        double bt = sqrt(gm1 * (gm1 + 2.0)) / (gm1 + 1.0);
        r_phi[index] += TWOPI * dd1 / (d_wavelen * bt); // absolute phase
      }
      else
      {
        double gam = r_design_w_in / d_const.mass;
        double beta_in = sqrt((gam + 2.0) * gam) / (gam + 1.0);
        gam = w / d_const.mass;
        double beta = sqrt((gam + 2.0) * gam) / (gam + 1.0);
        if(!flag_ccl) // dtl
        {
          double dps = gap.phase_ref - r_phi_in;
          double d_cell_len = cell_len - gap.beta_center * wave_len;
          double dphi_qlen1 = TWOPI * 0.5 * r_qlen1 / (beta * wave_len);
          double dphi_clen = 0.0;
          if(d_cell_len > 1e-15 || d_cell_len < -1e-15)
            dphi_clen = PI * d_cell_len / (beta * wave_len);
          r_phi[index] = phi + (1.0 - (beta - beta_in) / beta)*
            (PI*gap.cell_length_over_beta_lambda + dps) + dphi_clen - dphi_qlen1;
        }
        else // ccl
        {
          double dps = gap.phase_ref - r_phi_in;
          double betag = cell_len / (gap.cell_length_over_beta_lambda * 
	    wave_len);
          r_phi[index] = phi + (1.0 - (beta - beta_in) * betag / 
	    (beta * beta_in)) * (PI * gap.cell_length_over_beta_lambda + dps);
        }
      }
    }// if loss
    index += stride;
  } // while
}

__device__
void GetTransitTimeFactors(double* r_t, double* r_tp, double* r_sp, 
  double r_beta, double r_betag, double r_betamin, double r_ta0, double r_ta1, 
  double r_ta2, double r_ta3, double r_ta4, double r_ta5, double r_sa1, 
  double r_sa2, double r_sa3, double r_sa4, double r_sa5)
{
  double beta = r_beta;
  if (r_beta > 1.1 * r_betag)
    beta = r_betag;
  else if (r_beta < r_betamin)
    beta = r_betamin;
  *r_t = r_ta0 + beta * (r_ta1 + beta * (r_ta2 + beta * (r_ta3 + beta * 
    (r_ta4 + beta * r_ta5))));
  double coef = beta * beta / (TWOPI * r_betag); 
  *r_tp = coef * (r_ta1 + beta * (2*r_ta2 + beta * (3 * r_ta3 + beta * 
    (4 * r_ta4 + 5 * beta * r_ta5))));
  *r_sp = -coef * (r_sa1 + beta * (2*r_sa2 + beta * (3 * r_sa3 + beta * 
    (4 * r_sa4 + 5 * beta * r_sa5))));
}

__global__
void SimulateRFGapSecondHalfKernel(double* r_x, double* r_y, double* r_phi,
                            double* r_xp, double* r_yp, double* r_w, 
                            uint* r_loss, double r_design_w_in, 
                            double r_phi_in, RFGapParameter* r_elem, 
                            double r_length, uint r_ccl_cell_num, 
                            double r_qlen1 = 0.0, double r_qlen2 = 0.0,
                            bool flag_ccl = true) //, bool flag_horder_tf)
{
  RFGapParameter gap = *r_elem;
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  while(index < d_const.num_particle)
  { 
    if(r_loss[index] == 0)
    {
      double cell_len = r_length + r_qlen1 * 0.5 + r_qlen2 * 0.5;
      double dd2 = 0.5 * cell_len + gap.dg - r_qlen2 * 0.5;
      double x = r_x[index];
      double y = r_y[index];
      double xp = r_xp[index];
      double yp = r_yp[index];
      double inv_mass = 1.0 / d_const.mass;
      double phi = r_phi[index];
      double w = r_w[index];
      double gm_in = r_design_w_in * inv_mass + 1.0;
      double beta_in = sqrt(1.0 - 1.0 / (gm_in * gm_in));
      double gam = w * inv_mass;
      double bg1 = sqrt(gam * (gam + 2.0));    
      double beta = bg1 / (gam + 1.0);
      double t_fac = gap.t;
      double tp_fac = gap.tp;
      double sp_fac = gap.sp;
//      if (flag_horder_tf)
      GetTransitTimeFactors(&t_fac, &tp_fac, &sp_fac, beta, 
         gap.fit_beta_center, gap.fit_beta_min, 
         gap.fit_t0, gap.fit_t1, gap.fit_t2, gap.fit_t3, gap.fit_t4, gap.fit_t5,
         gap.fit_s1, gap.fit_s2, gap.fit_s3, gap.fit_s4, gap.fit_s5);

      if(gap.amplitude == 0.0 || (flag_ccl && beta < gap.fit_beta_min)) 
      {
        r_x[index] = x + dd2 * xp;
        r_y[index] = y + dd2 * yp;
        double gm1 = r_w[index] / d_const.mass;
        double bt = sqrt(gm1 * (gm1 + 2.0)) / (gm1 + 1.0);
        r_phi[index] += TWOPI * dd2 / (d_wavelen * bt); // absolute phase
      }
      else
      {
        double wave_len = CLIGHT / gap.frequency;
        double cl = cell_len / gap.cell_length_over_beta_lambda;
        double betag_ccl = cl / wave_len;
        double betag;
        if(flag_ccl) // ccl
          betag = gap.beta_center;// beta at the center of the gap
        else // dtl
          betag = betag_ccl;
        double betag2 = betag * betag;
        double gmg = 1.0 / sqrt(1.0 - betag2);
        double inv_gmg2 = 1.0 / (gmg * gmg);
        double q = d_const.charge;
        q = q > 0.0 ? q : -q;
        double gm_out = gap.energy_out * inv_mass + 1.0;
        double beta_out = sqrt(1.0 - 1.0 / (gm_out * gm_out));
        double efl = gap.amplitude * cell_len;
	// Full Bessel Function
        double concomm = q * efl / (d_const.mass * betag2);
        double concomm1 = concomm * inv_gmg2 / gmg;
        double con1 = PI * concomm1;
        double con2nex = concomm * inv_gmg2;
        double con5nex, tw, inv_fkis, inv_gmgcl;
        if(flag_ccl) //ccl
        {
          inv_gmgcl = 1.0 / (gmg * betag * wave_len);
          inv_fkis = betag * wave_len / TWOPI;
          con5nex = concomm * (t_fac * inv_fkis - betag_ccl * wave_len * 
	    tp_fac * inv_gmg2);
          tw = t_fac - TWOPI * tp_fac * (beta_in / beta - 1.0) * 
	    betag_ccl / beta_in;
        }
        else // dtl
        {
          inv_gmgcl = 1.0 / (gmg * cl);
          inv_fkis = cl / TWOPI;
          con5nex = concomm * (t_fac * inv_fkis - beta_in * wave_len * tp_fac 
	    * inv_gmg2);
          tw = t_fac;
        }
        double con5nex1 = concomm1 * t_fac;
        double con5nex2 = -con2nex * t_fac * inv_fkis;
        double fkr = TWOPI * inv_gmgcl;
        double sinps, cosps;
        sincos(gap.phase_ref, &sinps, &cosps);
        double phi1, sinp1, cosp1;
        if(flag_ccl) // ccl
          phi1 = phi + gap.phase_shift - TWOPI * gap.dg / (beta * wave_len) + 
	    r_ccl_cell_num * PI;
        else // dtl
          phi1 = phi + gap.phase_shift;
        sincos(phi1, &sinp1, &cosp1);
        double prome1 = -con1 * (tp_fac * (sinp1 - sinps) + sp_fac * 
	  (cosp1 - cosps));
        sincos(phi1 + prome1, &sinp1, &cosp1);

        double rs_p = x * x + y * y;
        double rrp = x * xp + y * yp;
        double ris = 0.0, fkrris = 0.0, rpris = 0.0, bes1covr = 0.0;
	// Full Bessel Function
        ris = sqrt(rs_p);
        fkrris = fkr * ris;
        double b0c = Bessi0c(fkrris);
        double b1c = Bessi1c(fkrris);
        double b1p = Bessi1p(fkrris);
        if(ris != 0.0)
          bes1covr = b1c / ris;
        else
          bes1covr = fkr * 0.5;
        double delte = q * efl * (tw * b0c * cosp1 + ((tw - TWOPI * tp_fac) /
	  gmg * bes1covr + tw / gmg * fkr * b1p - tw * gmg * bes1covr) * 
	  rrp * sinp1);
        w += delte;
        gam = w * inv_mass;
        double bg = sqrt(gam * (gam + 2.0));
        beta = bg / (gam + 1.0);
        double bgr = bg1 / bg;
	// Full Bessel Function
        double onema = 1.0 - (con5nex * bes1covr + con5nex1 * b1p
		      + con5nex2 * bes1covr) * cosp1;
        double dovbl = con2nex * bes1covr * tw * sinp1;
        double bgrovonema = bgr / onema;
        xp = -dovbl * x + bgrovonema * xp;
        x = x * onema + dd2 * xp;
        yp = -dovbl * y + bgrovonema * yp;
        y = y * onema + dd2 * yp;
        if(rs_p > 0.0)
          rpris = rrp / ris;

	// Full Bessel Function
        double prome = 2.0 * con1 * (-tp_fac * b0c + ris * tw * b1c *
	   inv_gmgcl) * (sinp1 - sinps) - efl * inv_mass * tw * b1c * rpris *
	   (cosp1 - cosps);

        double phi_new;
        if(flag_ccl) // ccl
        {
          double dps = r_phi_in - gap.phase_ref;
          phi_new = phi + (1.0 - (beta - beta_out) * betag_ccl / 
	    (beta * beta_out)) * (PI*gap.cell_length_over_beta_lambda + dps) 
	    + prome; // absolute phase
        }
        else // dtl
        {
          double d_cell_len = cell_len - gap.beta_center * wave_len;
          double dphi_qlen2 = TWOPI * 0.5 * r_qlen2 / (beta * wave_len);
          double dphi_clen = 0.0;
          if(d_cell_len > 1e-15 || d_cell_len < -1e-15)
            dphi_clen = PI * d_cell_len / (beta * wave_len);
          double dps = gap.phase_ref - r_phi_in;
          phi_new = phi + (1.0 - (beta - beta_out) / beta) * (PI * 
	    gap.cell_length_over_beta_lambda + dps) + prome + 
	    dphi_clen - dphi_qlen2; // absolute phase
        }

        if(w <= 0.0 || x != x || y != y || xp != xp || yp != yp || w != w || 
	  phi_new != phi_new)
        {
          r_loss[index] = 33333333; //TODO: change this.
          if (w > 0.0)
          {
            printf("NAN error, onema= %f, dovbl = %f\n", onema, dovbl);
            printf("NAN error, coordinates : %f, %f, %f, %f, %f, %f\n", x, xp, 
	      y, yp, r_phi[index], w);
          }
        }
        else
        {
          r_x[index] = x;
          r_y[index] = y;
          r_xp[index] = xp;
          r_yp[index] = yp;
          r_w[index] = w;
          r_phi[index] = phi_new;
        }
      }// if rf_amp == 0.0
    }// if loss
    index += stride;  
  }// while
}

__global__
void SimulateDisplaceKernel(double* r_x, double* r_y, uint* r_loss, double r_dx, double r_dy)
{
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index < d_const.num_particle && r_loss[index] == 0)
  {
    r_x[index] -= r_dx;
    r_y[index] -= r_dy;
  }
}

__global__
void SimulateRotationKernel(double* r_x, double* r_y, double* r_xp, double* r_yp, uint* r_loss, double r_angle)
{
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index < d_const.num_particle && r_loss[index] == 0)
  {
    double x = r_x[index];
    double y = r_y[index];
    double xp = r_xp[index];
    double yp = r_yp[index];
    double cosa = cos(r_angle);
    double sina = sin(r_angle);
    r_x[index] = x*cosa + y*sina;
    r_y[index] = y*cosa - x*sina;
    r_xp[index] = xp*cosa + yp*sina;
    r_yp[index] = yp*cosa - xp*sina;
  }
}

__global__
void SimulateTiltKernel(double* r_xp, double* r_yp, uint* r_loss, double r_dxp, double r_dyp)
{
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index < d_const.num_particle && r_loss[index] == 0)
  {
    r_xp[index] += r_dxp;
    r_yp[index] += r_dyp;
  }
}

__global__
void SimulateCircularApertureKernel(double* r_x, double* r_y, uint* r_loss, 
  double r_aper, double* r_center_x, double* r_center_y, int r_elem_indx)
{
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index > 0 && index < d_const.num_particle && r_loss[index] == 0)
  {
    double aper2 = r_aper*r_aper;
    double x = r_x[index] - r_center_x[0];
    double y = r_y[index] - r_center_y[0];
    if(x*x + y*y > aper2)
      r_loss[index] = r_elem_indx;
  }
}

__global__
void SimulateRectangularApertureKernel(double* r_x, double* r_y, uint* r_loss, 
  double r_aper_xl, double r_aper_xr, double r_aper_yt, double r_aper_yb, 
  double* r_center_x, double* r_center_y, int r_elem_indx)
{
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  if(index > 0 && index < d_const.num_particle && r_loss[index] == 0)
  {
    double xmin = r_center_x[0] - r_aper_xl; 
    double xmax = r_center_x[0] + r_aper_xr; 
    double ymin = r_center_y[0] - r_aper_yb;
    double ymax = r_center_y[0] + r_aper_yt; 
    double x = r_x[index];
    double y = r_y[index];
    if(x < xmin || x > xmax || y < ymin || y > ymax)
      r_loss[index] = r_elem_indx;
  }
}

__global__
void SetPlottingDataKernel(double* r_xavg_o, double* r_xavg,
                           double* r_yavg_o, double* r_yavg,
                           double* r_xsig_o, double* r_xsig,
                           double* r_ysig_o, double* r_ysig,
                           double* r_xemit_o, double* r_xemit,
                           double* r_yemit_o, double* r_yemit,
                           double* r_lemit_o, double* r_lemit,
                           double* r_num_loss_o, uint* r_num_loss,
                           uint r_index)
{
  r_xavg_o[r_index] = r_xavg[0]; 
  r_yavg_o[r_index] = r_yavg[0]; 
  r_xsig_o[r_index] = r_xsig[0]; 
  r_ysig_o[r_index] = r_ysig[0]; 
  r_xemit_o[r_index] = r_xemit[0]; 
  r_yemit_o[r_index] = r_yemit[0]; 
  r_lemit_o[r_index] = r_lemit[0]; 
  r_num_loss_o[r_index] = r_num_loss[0]; 
}

/*! The reason Dipole was not broken up into left edge + half sector + half sector 
+ right edge is to copy data once (x,y,xp,yp), do as many calc as it can. 
Breaking into smaller pieces needs more copying*/
__global__
void SimulateFirstHalfDipoleKernel(double* r_x, double* r_y, double* r_phi, 
  double* r_xp,double* r_yp, double* r_w, uint* r_loss, DipoleParameter* r_elem)
{
  DipoleParameter dipole = *r_elem;
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  uint stride = blockDim.x*gridDim.x;

  while(index < d_const.num_particle)
  {
    if(r_loss[index] == 0)
    {
      double x = r_x[index];
      double xp = r_xp[index];
      double y = r_y[index];
      double yp = r_yp[index];

      double h = 1.0/dipole.radius;
      double b1 = dipole.edge_angle_in;
      double hf_g = dipole.half_gap;
      double k1 = dipole.k1;
      double k2 = dipole.k2;
      double findx = dipole.field_index;
      double design_w = dipole.kinetic_energy;
      
      // first edge
      double c1 = 2.0*hf_g*k1*h;
      double sinb;// = sin(b1);
      double inv_cosb;// = 1.0/cos(b1);
      sincos(b1, &sinb, &inv_cosb);
      inv_cosb = 1.0/inv_cosb;
      double psi = c1*(1.0+sinb*sinb)*inv_cosb*(1.0-c1*k2*sinb*inv_cosb);
      double r21 = tan(b1)*h;
      double r43 = tan(b1-psi)*h;
      xp += r21*x;
      yp -= r43*y;
      // first half sector dipole
      double h2 = h*h;
      double dl = 0.5 * dipole.radius * dipole.angle;
      double yk2 = findx*h2;
      double yk = sqrt(abs(yk2));
      double inv_yk = 1.0/yk;
      double ykl = yk*dl;
      double xk2 = h2-yk2;
      double inv_xk2 = 1.0/xk2;
      double xk = sqrt(abs(xk2));
      double inv_xk = 1.0/xk;
      double xkl = xk*dl;
      double cxkl = 1.0;
      double sxokx = dl;
      if(xk2 > 0.0)
      {
        cxkl = cos(xkl);
        sxokx = sin(xkl)*inv_xk;
      }
      else if(xk2 < 0.0)
      {
        cxkl = cosh(xkl);
        sxokx = sinh(xkl)*inv_xk;
      }
      double cykl = 1.0;
      double syoky = dl;
      if(yk2 > 0.0)
      {
        cykl = cos(ykl);
        syoky = sin(ykl)*inv_yk;
      }
      else if(yk2 < 0.0)
      {
        cykl = cosh(ykl);
        syoky = sinh(ykl)*inv_yk;
      }
      double r11 = cxkl;
      double r12 = sxokx;
      r21 = -xk2*sxokx;
      double r22 = cxkl;
      double r33 = cykl;
      double r34 = syoky;
      r43 = -yk2*syoky;
      double r44 = cykl;
      double r26 = h*sxokx;
      double r16 = h*(1.0-cxkl)*inv_xk2;
      double r51 = -r26;
      double r52 = -r16;
      double gamma = design_w/d_const.mass + 1.0;
      double beta2 = 1.0-1.0/(gamma*gamma);
      double beta = sqrt(beta2);
      double r56 = -h2*(dl*beta2-sxokx)*inv_xk2+dl*(1.0-h2*inv_xk2)/(gamma*gamma);
      double dpop = gamma/(design_w*(gamma+1.0))*(r_w[index] - design_w);

      r_x[index] = r11*x + r12*xp + r16*dpop;
      r_xp[index] = r21*x + r22*xp + r26*dpop;
      r_y[index] = r33*y + r34*yp;
      r_yp[index] = r43*y + r44*yp;
      r_phi[index] += TWOPI*(-(r51*x + r52*xp + r56*dpop) + dl)/(beta*d_wavelen);
    }// if loss
    index += stride;
  }// while
}

__global__
void SimulateSecondHalfDipoleKernel(double* r_x, double* r_y, double* r_phi,
  double* r_xp, double* r_yp, double* r_w, uint* r_loss, DipoleParameter* r_elem)
{
  DipoleParameter dipole = *r_elem;
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  uint stride = blockDim.x*gridDim.x;
  while(index < d_const.num_particle)
  {
    if(r_loss[index] == 0)
    {
      double x = r_x[index];
      double xp = r_xp[index];
      double y = r_y[index];
      double yp = r_yp[index];

      double h = 1.0/dipole.radius;
      double b2 = dipole.edge_angle_out;
      double hf_g = dipole.half_gap;
      double k1 = dipole.k1;
      double k2 = dipole.k2;
      double findx = dipole.field_index;
      double design_w = dipole.kinetic_energy;

      // second half sector dipole
      double h2 = h*h;
      double dl = 0.5 * dipole.radius * dipole.angle;
      double yk2 = findx*h2;
      double yk = sqrt(abs(yk2));
      double inv_yk = 1.0/yk;
      double ykl = yk*dl;
      double xk2 = h2-yk2;
      double inv_xk2 = 1.0/xk2;
      double xk = sqrt(abs(xk2));
      double inv_xk = 1.0/xk;
      double xkl = xk*dl;
      double cxkl = 1.0;
      double sxokx = dl;
      if(xk2 > 0.0)
      {
        cxkl = cos(xkl);
        sxokx = sin(xkl)*inv_xk;
      }
      else if(xk2 < 0.0)
      {
        cxkl = cosh(xkl);
        sxokx = sinh(xkl)*inv_xk;
      }
      double cykl = 1.0;
      double syoky = dl;
      if(yk2 > 0.0)
      {
        cykl = cos(ykl);
        syoky = sin(ykl)*inv_yk;
      }
      else if(yk2 < 0.0)
      {
        cykl = cosh(ykl);
        syoky = sinh(ykl)*inv_yk;
      }
      double r11 = cxkl;
      double r12 = sxokx;
      double r21 = -xk2*sxokx;
      double r22 = cxkl;
      double r33 = cykl;
      double r34 = syoky;
      double r43 = -yk2*syoky;
      double r44 = cykl;
      double r26 = h*sxokx;
      double r16 = h*(1.0-cxkl)*inv_xk2;
      double r51 = -r26;
      double r52 = -r16;
      double gamma = design_w/d_const.mass + 1.0;
      double beta2 = 1.0-1.0/(gamma*gamma);
      double beta = sqrt(beta2);
      double r56 = -h2*(dl*beta2-sxokx)*inv_xk2+dl*(1.0-h2*inv_xk2)/(gamma*gamma);
      double dpop = gamma/(design_w*(gamma+1.0))*(r_w[index] - design_w);
      r_phi[index] += TWOPI*(-(r51*x + r52*xp + r56*dpop) + dl)/(beta*d_wavelen);
      double x_new = r11*x + r12*xp + r16*dpop;
      xp = r21*x + r22*xp + r26*dpop;
      double y_new = r33*y + r34*yp;
      yp = r43*y + r44*yp;
      r_x[index] = x_new;
      r_y[index] = y_new;
      // second edge
      double c1 = 2.0*hf_g*k1*h;
      double sinb = sin(b2);
      double inv_cosb = 1.0/cos(b2);
      double psi = c1*(1.0+sinb*sinb)*inv_cosb*(1.0-c1*k2*sinb*inv_cosb);
      r21 = tan(b2)*h;
      r43 = tan(b2-psi)*h;
      r_xp[index] = xp + r21*x_new;
      r_yp[index] = yp - r43*y_new;
    }// if loss
    index += stride;
  }// while
}

__global__
void SimulateBuncherKernel(double* r_x, double* r_y, double* r_phi, 
  double* r_xp, double* r_yp, double* r_w, uint* r_loss, double r_w_ref,
  double r_freq, double r_voltage, double r_phase, double r_aper, 
  double r_freq_ratio, uint r_elem_indx)
{
  uint index = blockIdx.x*blockDim.x+threadIdx.x;
  uint stride = blockDim.x*gridDim.x;
  while(index < d_const.num_particle)
  {
    if(r_loss[index] == 0)
    {
      double x = r_x[index];
      double y = r_y[index];
      double xp = r_xp[index];
      double yp = r_yp[index];
      double phi = r_phi[index] * r_freq_ratio;
      double w = r_w[index];
      double r2 = x*x + y*y;
      double aper2 = r_aper * r_aper;
      // check for loss
      if(aper2 != 0.0 && r2 > aper2)
      {
        r_loss[index] = r_elem_indx;
        index += stride;
        continue;
      }
      double h = r_freq*d_wavelen/CLIGHT;
      double gm1 = r_w_ref/d_const.mass;
      double gm = gm1 + 1.0;
      double btgm = sqrt(gm1*(gm1+2.0)); 
      double beta = btgm/gm; 
      double k = h*TWOPI/(btgm*d_wavelen); 
      double k2 = k*k; 
      double q = d_const.charge;
      q = q > 0.0 ? q : -q;
      double con1 = TWOPI*h*r_voltage*q/(beta*d_wavelen*d_const.mass);
      double arg = k2*r2*0.25;
      double ph = h*phi + r_phase;
      double dw = q*r_voltage*cos(ph)*(1.0+arg*(1.0+arg*(0.25+arg/36.0)));
      double wb_par = w + 0.5*dw;
      double wf = w + dw;
      if(wb_par <= 0.0 || wf <= 0.0)
      {
        r_loss[index] = r_elem_indx;
        index += stride;
        continue;
      }
      double bav, gmtmp, bg, rbgf, del;
      gmtmp = wb_par/d_const.mass;
      bav = sqrt(gmtmp*(gmtmp+2.0))/(gmtmp+1.0);
      gmtmp = w/d_const.mass;
      bg = sqrt(gmtmp*(gmtmp+2.0));
      gmtmp = wf/d_const.mass;
      rbgf = rsqrt(gmtmp*(gmtmp+2.0));
      del = -con1*sin(ph)*(1.0-bav*beta)*(0.5+arg*(0.25+arg/24.0))/bav;
      r_xp[index] = (xp*bg + del*x)*rbgf;
      r_yp[index] = (yp*bg + del*y)*rbgf;
      r_w[index] = wf;
    }
    index += stride;
  }
}

#endif
