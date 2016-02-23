#ifndef RFGAP_PARAMETER_H
#define RFGAP_PARAMETER_H
  
struct RFGapParameter
{
  double length;
  double frequency;
  double cell_length_over_beta_lambda;
  double amplitude;
  double phase_ref;
  double phase_shift;
  double energy_out;
  double beta_center;
  double dg;
  double t;
  double tp;
  double sp;
  // For higher order transit time factor 
  double fit_beta_center;
  double fit_beta_min;
  double fit_t0;
  double fit_t1;
  double fit_t2;
  double fit_t3;
  double fit_t4;
  double fit_t5;
  double fit_s1;
  double fit_s2;
  double fit_s3;
  double fit_s4;
  double fit_s5;
};

#endif
