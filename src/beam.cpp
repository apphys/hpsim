#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>
#include <cctype>
#include "beam.h"
#include "beam_cu.h"
#include "utility.h"

Beam::Beam() : PyWrapper(), beam_0(std::vector<double*>(6, NULL)), loss_0(NULL),
  lloss_0(NULL), beam_1(std::vector<double*>(6, NULL)), loss_1(NULL), 
  lloss_1(NULL), freq(0.0)
{
}

void Beam::AllocateBeam(uint r_num, double r_mass, double r_charge, double r_current)
{
  num_particle = r_num;
  mass = r_mass;
  charge = r_charge;
  current = r_current;
  CreateBeamOnDevice(this);
  ConfigureThreads(256);
  InitPartials();
}

void Beam::SaveBeam(std::vector<double*>& r_beam, uint*& r_loss, uint*& r_lloss)
{
  for(int i = 0; i < r_beam.size(); ++i)
  {
    if(r_beam[i] == NULL)
      r_beam[i] = new double[num_particle];
  } 
  if(r_loss == NULL)
    r_loss = new uint[num_particle];
  if(r_lloss == NULL)
    r_lloss = new uint[num_particle];

  uint* lnum = new uint;
  CopyBeamFromDevice(this, r_beam[0], r_beam[1], r_beam[2], r_beam[3], 
    r_beam[4], r_beam[5], r_loss, r_lloss, lnum); 
  delete lnum;
}

void Beam::SaveInitialBeam()
{
  SaveBeam(beam_0, loss_0, lloss_0);
  design_w_0 = design_w;
  freq_0 = freq;
}

void Beam::SaveIntermediateBeam()
{
  SaveBeam(beam_1, loss_1, lloss_1);
  design_w_1 = design_w;
  freq_1 = freq;
}

void Beam::RestoreBeam(std::vector<double*>& r_beam, uint*& r_loss, uint*& r_lloss)
{
  if(r_loss == NULL)
  {
    std::cerr << "Beam can't be restored because no previous beam has been saved. " << std::endl;
    return;
  } 
  UpdateBeamOnDevice(this, r_beam[0], r_beam[1], r_beam[2], r_beam[3], 
    r_beam[4], r_beam[5], r_loss, r_lloss);
}

void Beam::RestoreInitialBeam()
{
  RestoreBeam(beam_0, loss_0, lloss_0);
  design_w = design_w_0;
  freq = freq_0;
  UpdateAvgPhi(true);
  UpdateRelativePhi(true);
}

void Beam::RestoreIntermediateBeam()
{
  RestoreBeam(beam_1, loss_1, lloss_1); 
  design_w = design_w_1;
  freq = freq_1;
  UpdateAvgPhi(true);
  UpdateRelativePhi(true);
}

void Beam::InitPhiAvgGood()
{
  UpdateLoss();
  UpdateAvgPhi();
//  InitPhiAvg(this);
  CopyVariable(&phi_avg_good, &phi_avg);
//  UpdateRelativePhi(true);
//  std::cout << "@@@@@ Beam::InitAvgPhiGood : " << GetDataFromDevice(phi_avg, 0) 
//    << ", " << GetDataFromDevice(phi_avg_good, 0) << std::endl;
}

void Beam::InitBeamFromDistribution(std::vector<double>& r_x, 
  std::vector<double>& r_xp, std::vector<double>& r_y, 
  std::vector<double>& r_yp, std::vector<double>& r_phi,
  std::vector<double>& r_w, std::vector<uint>* r_loss, 
  std::vector<uint>* r_lloss)
{
  if(r_x.size() != num_particle)
  {
    std::cerr << "Beam::InitBeamFromDistribution error: coordinate size != "
        "num_particle." << std::endl; 
    exit(0);
  }
  double* x_h = new double[num_particle];
  double* xp_h = new double[num_particle];
  double* y_h = new double[num_particle];
  double* yp_h = new double[num_particle];
  double* phi_h = new double[num_particle];
  double* w_h = new double[num_particle];
  std::copy(r_x.begin(), r_x.end(), x_h);
  std::copy(r_xp.begin(), r_xp.end(), xp_h);
  std::copy(r_y.begin(), r_y.end(), y_h);
  std::copy(r_yp.begin(), r_yp.end(), yp_h);
  std::copy(r_phi.begin(), r_phi.end(), phi_h);
  std::copy(r_w.begin(), r_w.end(), w_h);
  x_h[0] = 0.0; xp_h[0] = 0.0;
  y_h[0] = 0.0; yp_h[0] = 0.0;
  uint* loss_h = NULL, *lloss_h = NULL;
  if(r_loss != NULL)
  {
    loss_h = new uint[num_particle];
    std::copy(r_loss->begin(), r_loss->end(), loss_h);
    loss_h[0] = 0;
  }
  if(r_lloss != NULL)
  {
    lloss_h = new uint[num_particle];
    std::copy(r_lloss->begin(), r_lloss->end(), lloss_h);
    lloss_h[0] = 0; // ???
  }
  UpdateBeamOnDevice(this, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h); 
  if(loss_h != NULL)
    delete [] loss_h;
  if(lloss_h != NULL)
    delete [] lloss_h;
  delete [] x_h; delete [] xp_h; 
  delete [] y_h; delete [] yp_h;
  delete [] phi_h; delete [] w_h;
  InitPhiAvgGood();
}

void Beam::InitBeamFromFile(std::string r_file)
{
  std::ifstream input(r_file.c_str());
  std::string tmp;
  int par_cnt = 0;
  bool charge_not_found = true;
  bool mass_not_found = true;
  bool current_not_found = true;
  bool freq_not_found = true;
  while(std::getline(input, tmp))
  {
    if(MatchCaseInsensitive(tmp, "charge") && ContainNumbers(tmp))
    {
      charge_not_found = false;
      charge = GetFirstNumberInString(tmp);
    }
    if(MatchCaseInsensitive(tmp, "mass") && ContainNumbers(tmp))
    {
      mass_not_found = false;
      mass = GetFirstNumberInString(tmp);
    }
    if(MatchCaseInsensitive(tmp, "current") && ContainNumbers(tmp))
    {
      current_not_found = false;
      current = GetFirstNumberInString(tmp);
    }
    if(MatchCaseInsensitive(tmp, "frequency") && ContainNumbers(tmp))
    {
      freq_not_found = false;
      freq = GetFirstNumberInString(tmp);
    }

    if(ContainOnlyNumbers(tmp))
      par_cnt++;
  }
  if(charge_not_found)
  {
    std::cerr << "Beam input file error: must define charge value! " << std::endl;
    exit(-1);
  }
  if(mass_not_found)
  {
    std::cerr << "Beam input file error: must define mass value! " << std::endl;
    exit(-1);
  }
  if(current_not_found)
  {
    std::cerr << "Beam input file error: must define current value! " << std::endl;
    exit(-1);
  }
  if(freq_not_found)
  {
    std::cerr << "Beam input file error: must define frequency value! " << std::endl;
    exit(-1);
  }
  input.close(); 
  AllocateBeam(par_cnt, mass, charge, current);
  UpdateBeamFromFile(r_file);
  InitPhiAvgGood();
}

void Beam::InitBeamFromFile(std::string r_file, double r_mass, double r_charge, 
                            double r_current)
{
  mass = r_mass;
  charge = r_charge;
  current = r_current;
  std::ifstream input(r_file.c_str());
  std::string tmp;
  int cnt = 0;
  while(std::getline(input, tmp)) 
  {
    if(tmp != "")
      ++cnt;
  }
  input.close();
  AllocateBeam(cnt, r_mass, r_charge, r_current);
  UpdateBeamFromFile(r_file); 
  InitPhiAvgGood();
}
/*
 * r_ax in radian
 * r_bx in cm
 * r_ex in cm*radian
 * r_sync_phi in radian
 * r_w in MeV
 */
void Beam::InitWaterbagBeam(double r_ax, double r_bx, double r_ex,
  double r_ay, double r_by, double r_ey, double r_az, double r_bz, double r_ez,
  double r_sync_phi, double r_sync_w, double r_freq, uint r_seed)
{
  double* x_h = new double[num_particle];
  double* xp_h = new double[num_particle];
  double* y_h = new double[num_particle];
  double* yp_h = new double[num_particle];
  double* phi_h = new double[num_particle];
  double* w_h = new double[num_particle];

  if(r_seed == 0)
    std::srand(std::time(NULL));
  else 
    std::srand(r_seed);
  
  double rsqrt = 1.0;//std::sqrt(1.0/6.0);
  double gx = (1.0 + r_ax*r_ax)/r_bx;
  double xmx = std::sqrt(r_ex/gx);
  double xpmx = std::sqrt(r_ex*gx);
  double dx = -r_ax/gx;
  double gy = (1.0 + r_ay*r_ay)/r_by;
  double ymx = std::sqrt(r_ey/gy);
  double ypmx = std::sqrt(r_ey*gy);
  double dy = -r_ay/gy;
  double gz = (1.0 + r_az*r_az)/r_bz;
  double zmx = std::sqrt(r_ez/gz);
  double zpmx = r_ez/zmx;
  double dz = -r_az/gz;
  double inpde = 2.0*r_sync_w*zpmx;
  double wave_len = 29979.2458/r_freq; // in cm
  freq = r_freq;
  double gm = r_sync_w/mass + 1.0;
  double beta = std::sqrt(1.0-1.0/(gm*gm));

  for(int i = 0; i < num_particle; ++i)
  {
    double r1 = 1.0, r2 = 1.0, r3 = 1.0, r4 = 1.0, r5 = 1.0, r6 = 1.0;
    do{
      do{
        r1 = 2.0*std::rand()/RAND_MAX-1.0;
        r2 = 2.0*std::rand()/RAND_MAX-1.0;
      }while(r1*r1 + r2*r2 > 1.0);
      
      do{
        r3 = 2.0*std::rand()/RAND_MAX-1.0;
        r4 = 2.0*std::rand()/RAND_MAX-1.0;
      }while(r3*r3 + r4*r4 > 1.0);
      do{
        r5 = 2.0*std::rand()/RAND_MAX-1.0;
        r6 = 2.0*std::rand()/RAND_MAX-1.0;
      }while(r5*r5 + r6*r6 > 1.0);
    }while(r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6> 1.0);

    r1 *= rsqrt; r2 *= rsqrt; r3 *= rsqrt; r4 *= rsqrt; r5 *= rsqrt; r6 *= rsqrt;
    x_h[i] = 0.01*(r1*xmx + dx*r2*xpmx);
    xp_h[i] = r2*xpmx;
    y_h[i] = 0.01*(r3*ymx + dy*r4*ypmx);
    yp_h[i] = r4*ypmx;
    phi_h[i] = -(r5*zmx+dz*zpmx*r6)*2.0*M_PI/(beta*wave_len) + r_sync_phi;
    w_h[i] = 2.0*r_sync_w*zpmx*r6 + r_sync_w;
  }
  x_h[0] = 0.0; xp_h[0] = 0.0; y_h[0] = 0.0; yp_h[0] = 0.0;
  UpdateBeamOnDevice(this, x_h, xp_h, y_h, yp_h, phi_h, w_h);
  delete [] x_h; delete [] xp_h;
  delete [] y_h; delete [] yp_h;
  delete [] phi_h; delete [] w_h;
  InitPhiAvgGood();
}
 
/*
 * r_ax in radian
 * r_bx in cm
 * r_ex in cm*radian
 * r_sync_phi in radian
 * r_w in MeV
 */
void Beam::InitDCBeam(double r_ax, double r_bx, double r_ex,
                  double r_ay, double r_by, double r_ey, double r_dphi,
                  double r_sync_phi, double r_sync_w, uint r_seed)
{
  double* x_h = new double[num_particle];
  double* xp_h = new double[num_particle];
  double* y_h = new double[num_particle];
  double* yp_h = new double[num_particle];
  double* phi_h = new double[num_particle];
  double* w_h = new double[num_particle];

  if(r_seed == 0)
    std::srand(std::time(NULL));
  else 
    std::srand(r_seed);
  
  double rsqrt = 1.0;//std::sqrt(1.0/6.0);
  double gx = (1.0 + r_ax*r_ax)/r_bx;
  double xmx = std::sqrt(r_ex/gx);
  double xpmx = std::sqrt(r_ex*gx);
  double dx = -r_ax/gx;
  double gy = (1.0 + r_ay*r_ay)/r_by;
  double ymx = std::sqrt(r_ey/gy);
  double ypmx = std::sqrt(r_ey*gy);
  double dy = -r_ay/gy;
  for(int i = 0; i < num_particle; ++i)
  {
    double r1 = 1.0, r2 = 1.0, r3 = 1.0, r4 = 1.0, r5 = 1.0;
    do{
      int cnt = 0;
      do{
        r1 = 2.0*std::rand()/RAND_MAX-1.0;
        r2 = 2.0*std::rand()/RAND_MAX-1.0;
      }while(r1*r1 + r2*r2 > 1.0);
      
      do{
        r3 = 2.0*std::rand()/RAND_MAX-1.0;
        r4 = 2.0*std::rand()/RAND_MAX-1.0;
      }while(r3*r3 + r4*r4 > 1.0);
    }while(r1*r1 + r2*r2 + r3*r3 + r4*r4 > 1.0);

    r1 *= rsqrt; r2 *= rsqrt; r3 *= rsqrt; r4 *= rsqrt;
    r5 = 2.0*(double)(i+1.0)/(double)num_particle-1.0; 
    x_h[i] = 0.01*(r1*xmx + dx*r2*xpmx);
    xp_h[i] = r2*xpmx;
    y_h[i] = 0.01*(r3*ymx + dy*r4*ypmx);
    yp_h[i] = r4*ypmx;
    phi_h[i] = r5*r_dphi + r_sync_phi;
    w_h[i] = r_sync_w;
  }
  x_h[0] = 0.0; xp_h[0] = 0.0; y_h[0] = 0.0; yp_h[0] = 0.0;
  UpdateBeamOnDevice(this, x_h, xp_h, y_h, yp_h, phi_h, w_h);
  delete [] x_h; delete [] xp_h;
  delete [] y_h; delete [] yp_h;
  delete [] phi_h; delete [] w_h;
  InitPhiAvgGood();
}

void Beam::UpdateBeamFromFile(std::string r_file)
{
  double* x_h = new double[num_particle];
  double* xp_h = new double[num_particle];
  double* y_h = new double[num_particle];
  double* yp_h = new double[num_particle];
  double* phi_h = new double[num_particle];
  double* w_h = new double[num_particle];
  uint* loss_h = new uint[num_particle];
  uint* lloss_h = new uint[num_particle];
  int cnt = 0; 
  int num_col = 6;
  std::string tmp;
  std::ifstream input(r_file.c_str());
  while(std::getline(input, tmp)) 
  {
    if(tmp == "" || !(ContainOnlyNumbers(tmp))) continue;
    std::istringstream iss(tmp);
    std::vector<std::string> tmp_vec;
    tmp_vec.assign(std::istream_iterator<std::string>(iss), 
                   std::istream_iterator<std::string>());
    x_h[cnt] = std::atof(tmp_vec[0].c_str()); 
    xp_h[cnt] = std::atof(tmp_vec[1].c_str());
    y_h[cnt] = std::atof(tmp_vec[2].c_str()); 
    yp_h[cnt] = std::atof(tmp_vec[3].c_str());
    phi_h[cnt] = std::atof(tmp_vec[4].c_str()); 
    w_h[cnt] = std::atof(tmp_vec[5].c_str());
    num_col = tmp_vec.size();
    if (tmp_vec.size() > 6)
      loss_h[cnt] = std::atoi(tmp_vec[6].c_str()); 
    if (tmp_vec.size() > 7) 
      lloss_h[cnt] = std::atoi(tmp_vec[7].c_str()); 
    ++cnt;
  }
  input.close();
  x_h[0] = 0.0; xp_h[0] = 0.0; y_h[0] = 0.0; yp_h[0] = 0.0;
  design_w = w_h[0];
  if(num_col > 7)
    UpdateBeamOnDevice(this, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h);
  else if(num_col > 6)
    UpdateBeamOnDevice(this, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h);
  else
    UpdateBeamOnDevice(this, x_h, xp_h, y_h, yp_h, phi_h, w_h);
  delete [] x_h; delete [] xp_h;
  delete [] y_h; delete [] yp_h;
  delete [] phi_h; delete [] w_h; 
  delete [] loss_h; delete [] lloss_h;
}

void Beam::ConfigureThreads(uint r_blck_size)
{
  uint num_thread = NextPow2(num_particle);
  if(num_thread > r_blck_size)
  {
    grid_size = num_thread/r_blck_size;
    blck_size = r_blck_size;
  }
  else
  {
    grid_size = 1;
    blck_size = num_thread;
  }
}

void Beam::InitPartials()
{
  CreatePartialsOnDevice(this, grid_size, blck_size);
}

void Beam::FreeBeam()
{
  FreeBeamOnDevice(this);
  for(int i = 0; i < beam_0.size(); ++i)
  {
    if(beam_0[i] != NULL)
      delete [] beam_0[i]; 
    if(beam_1[i] != NULL)
      delete [] beam_1[i]; 
  } 
  if(loss_0 != NULL)
    delete [] loss_0;
  if(lloss_0 != NULL)
    delete [] lloss_0;
  if(loss_1 != NULL)
    delete [] loss_1;
  if(lloss_1 != NULL)
    delete [] lloss_1;
  std::cout << "Beam is freed. " << std::endl;
}

void Beam::Print(std::string r_file, std::string r_msg)
{
  UpdateLoss();
  const uint num = num_particle;
  double* x_h = new double[num];
  double* xp_h = new double[num];
  double* y_h = new double[num];
  double* yp_h = new double[num];
  double* phi_h = new double[num];
  double* w_h = new double[num];
  uint* loss_h = new uint[num];
  uint* lloss_h = new uint[num];
  uint* num_loss_h = new uint;
  CopyBeamFromDevice(this, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h, num_loss_h);
  std::ofstream out(r_file.c_str());
  out << "Info " << r_msg << std::endl;
  out << "Charge  " << charge << std::endl;
  out << "Mass " << mass << std::endl;
  out << "Current " << current << std::endl;
  out << "Frequency " << freq << std::endl;
  out << std::setprecision(15);
//  out << "Loss Num: " << *num_loss_h << std::endl;
  for(int i = 0; i < num; ++i)
    out << x_h[i] << "  " << xp_h[i] << "  "
              << y_h[i] << "  " << yp_h[i] << "  " << phi_h[i] << "  "
              << w_h[i] << "  " << loss_h[i] << "  " << lloss_h[i] << std::endl;
  out.close();
  delete [] x_h;
  delete [] xp_h;
  delete [] y_h;
  delete [] yp_h;
  delete [] phi_h;
  delete [] w_h;
  delete [] loss_h;
  delete [] lloss_h;
  delete num_loss_h;
}

void Beam::PrintSimple()
{
  std::cout << "Beam, mass = " << mass << ", charge = " << charge << ", current = " << current << std::endl;
  Print(0);
  Print(num_particle/2-1);
  Print(num_particle-2);
}

void Beam::Print(uint r_indx)
{
  double* x_h = new double;
  double* xp_h = new double;
  double* y_h = new double;
  double* yp_h = new double;
  double* phi_h = new double;
  double* w_h = new double;
  uint* loss_h = new uint;
  uint* lloss_h = new uint;
  CopyParticleFromDevice(this, r_indx, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h);
  std::cout << std::setprecision(15) << std::fixed;
  std::cout << *x_h << "\t" << *xp_h << "\t" << *y_h << "\t" << *yp_h << "\t" 
    << *phi_h << "\t" << *w_h << "\t" << *loss_h << "\t" << *lloss_h << std::endl;
  delete x_h;
  delete xp_h;
  delete y_h;
  delete yp_h;
  delete phi_h;
  delete w_h;
  delete loss_h;
  delete lloss_h;
}

void Beam::UpdateLoss()
{
  UpdateLossCountKernelCall(this);
}

void Beam::UpdateLongitudinalLoss()
{
  UpdateGoodParticleCount();
  UpdateAvgPhi(true);
  UpdateLongitudinalLossCoordinateKernelCall(this);
  UpdateLossCountKernelCall(this, true);
}

void Beam::UpdateGoodParticleCount()
{
  UpdateGoodParticleCountKernelCall(this);
}

void Beam::UpdateAvgXp()
{
  UpdateAvgOfOneVariableKernelCall(this, xp, xp_avg);
}
void Beam::UpdateAvgYp()
{
  UpdateAvgOfOneVariableKernelCall(this, yp, yp_avg);
}

void Beam::UpdateAvgX(bool r_good_only)
{
  if (r_good_only)
    UpdateAvgOfOneVariableKernelCall(this, x, x_avg_good, true);
  else
    UpdateAvgOfOneVariableKernelCall(this, x, x_avg);
}

void Beam::UpdateAvgY(bool r_good_only)
{
  if (r_good_only)
    UpdateAvgOfOneVariableKernelCall(this, y, y_avg_good, true);
  else
    UpdateAvgOfOneVariableKernelCall(this, y, y_avg);
}

void Beam::UpdateAvgPhi(bool r_good_only)
{
  if (r_good_only)
    UpdateAvgOfOneVariableKernelCall(this, phi, phi_avg_good, true);
  else
    UpdateAvgOfOneVariableKernelCall(this, phi, phi_avg);
}

void Beam::UpdateAvgRelativePhi(bool r_good_only)
{
  if (r_good_only)
    UpdateAvgOfOneVariableKernelCall(this, phi_r, phi_avg_r, true);
  else
    UpdateAvgOfOneVariableKernelCall(this, phi_r, phi_avg_r);
}

void Beam::UpdateAvgW(bool r_good_only)
{
  if (r_good_only)
    UpdateAvgOfOneVariableKernelCall(this, w, w_avg_good, true);
  else
    UpdateAvgOfOneVariableKernelCall(this, w, w_avg);
}

void Beam::UpdateSigXp()
{
  if(num_particle > 2)
    UpdateSigmaOfOneVariableKernelCall(this, xp, xp_sig);
}

void Beam::UpdateSigYp()
{
  if(num_particle > 2)
    UpdateSigmaOfOneVariableKernelCall(this, yp, yp_sig);
}

void Beam::UpdateSigX(bool r_good_only)
{
  if(num_particle > 2)
    if (r_good_only)
      UpdateSigmaOfOneVariableKernelCall(this, x, x_sig_good, r_good_only);
    else
      UpdateSigmaOfOneVariableKernelCall(this, x, x_sig);
}
void Beam::UpdateSigY(bool r_good_only)
{
  if(num_particle > 2)
    if (r_good_only)
      UpdateSigmaOfOneVariableKernelCall(this, y, y_sig_good, r_good_only);
    else
      UpdateSigmaOfOneVariableKernelCall(this, y, y_sig);
}
 
void Beam::UpdateSigPhi()
{
  if(num_particle > 2)
    UpdateSigmaOfOneVariableKernelCall(this, phi, phi_sig);
}
void Beam::UpdateSigRelativePhi(bool r_good_only)
{
  if(num_particle > 2)
    if(r_good_only)
      UpdateSigmaOfOneVariableKernelCall(this, phi_r, phi_sig_good, r_good_only);
    else
      UpdateSigmaOfOneVariableKernelCall(this, phi_r, phi_sig_r, r_good_only);
}

void Beam::UpdateSigW(bool r_good_only)
{
  if(num_particle > 2)
    if (r_good_only)
      UpdateSigmaOfOneVariableKernelCall(this, w, w_sig_good, r_good_only);
    else
      UpdateSigmaOfOneVariableKernelCall(this, w, w_sig);
}

//void Beam::UpdateSigR()
//{
//  UpdateSigmaR(this);
//}

void Beam::UpdateEmittance()
{
  if(num_particle > 2)
  {
    UpdateHorizontalEmittanceKernelCall(this);  
    UpdateVerticalEmittanceKernelCall(this);  
    UpdateRelativePhi();
    UpdateLongitudinalEmittanceKernelCall(this);  
  }
}

/*!
 * \brief Update averages of x, y, and phi and sigma of x, y
 */
void Beam::UpdateStatForSpaceCharge()
{
  if(num_particle > 2)
  {
//    UpdateLongitudinalLoss();
//    UpdateGoodParticleCount();
//    UpdateAvgPhi(true);
//    UpdateRelativePhi(true);
//    UpdateAvgOfOneVariableKernelCall(this, phi_r, phi_avg_r);
//    UpdateSigRelativePhi(true);  

    UpdateLongitudinalLoss();
    UpdateGoodParticleCount();
    UpdateAvgPhi(true);
    UpdateRelativePhi(true);
    UpdateAvgOfOneVariableKernelCall(this, phi_r, phi_avg_r);
    UpdateSigRelativePhi(true);
    UpdateAvgSigXYKernelCall(this);  
//    UpdateSigX(true);
//    UpdateSigY(true);
////    UpdateAvgX(true);
////    UpdateAvgY(true);
    UpdateAvgW(true);
  }
}
 
void Beam::UpdateStatForPlotting()
{
  UpdateStatForSpaceCharge();
  UpdateSigW(true);
  UpdateAvgRelativePhi(true);
}
/*!
 * \brief Used by rectangular aperture beamline element
 */
void Beam::UpdateAvgXY()
{
  UpdateAvgXYKernelCall(this);
}

void Beam::UpdateMaxR2Phi()
{
  UpdateMaxR2PhiKernelCall(this);
}

void Beam::UpdateRelativePhi(bool r_use_good)
{
  UpdateRelativePhiKernelCall(this, r_use_good);
}

void Beam::SetRefEnergy(double r_energy)
{
  SetDoubleValue(w, 0, r_energy);
  design_w = r_energy;
}

void Beam::SetRefPhase(double r_phase)
{
  SetDoubleValue(phi, 0, r_phase);
}

double Beam::GetRefEnergy() const
{
  return GetDataFromDevice(w, 0);
}

double Beam::GetMaxPhi() const
{
  return GetDataFromDevice(abs_phi_max, 0);
}


double Beam::GetRefPhase() const
{
  return GetDataFromDevice(phi, 0);
}

std::vector<double> Beam::GetRelativePhi() const
{
  return GetArrayFromDevice(phi_r, num_particle);
}

std::vector<double> Beam::GetX() const
{
  return GetArrayFromDevice(x, num_particle);
}

std::vector<double> Beam::GetXp() const
{
  return GetArrayFromDevice(xp, num_particle);
}

std::vector<double> Beam::GetY() const
{
  return GetArrayFromDevice(y, num_particle);
}

std::vector<double> Beam::GetYp() const
{
  return GetArrayFromDevice(yp, num_particle);
}

std::vector<double> Beam::GetPhi() const
{
  return GetArrayFromDevice(phi, num_particle);
}

std::vector<double> Beam::GetW() const
{
  return GetArrayFromDevice(w, num_particle);
}

std::vector<uint> Beam::GetLoss() const
{
  return GetArrayFromDevice(loss, num_particle);
}

std::vector<uint> Beam::GetLongitudinalLoss() const
{
  return GetArrayFromDevice(lloss, num_particle);
}

double Beam::GetAvgX(bool r_good_only) const
{
  if (r_good_only)
    return GetDataFromDevice(x_avg_good, 0);
  else
    return GetDataFromDevice(x_avg, 0);
}

double Beam::GetAvgY(bool r_good_only) const
{
  if (r_good_only)
    return GetDataFromDevice(y_avg_good, 0);
  else
    return GetDataFromDevice(y_avg, 0);
}

double Beam::GetAvgPhi(bool r_good_only) const
{
  if (r_good_only)
    return GetDataFromDevice(phi_avg_good, 0);
  else
    return GetDataFromDevice(phi_avg, 0);
}

double Beam::GetAvgRelativePhi() const
{
  return GetDataFromDevice(phi_avg_r, 0);
}

double Beam::GetAvgW(bool r_good_only) const
{
  if (r_good_only)
    return GetDataFromDevice(w_avg_good, 0);
  else
    return GetDataFromDevice(w_avg, 0);
}

double Beam::GetSigX(bool r_good_only) const
{
  if(num_particle < 2)
    return 0.0;
  if (r_good_only)
    return GetDataFromDevice(x_sig_good, 0);
  else
    return GetDataFromDevice(x_sig, 0);
}

double Beam::GetSigY(bool r_good_only) const
{
  if(num_particle < 2)
    return 0.0;
  if (r_good_only)
    return GetDataFromDevice(y_sig_good, 0);
  else
    return GetDataFromDevice(y_sig, 0);
}

double Beam::GetSigPhi() const
{
  if(num_particle < 2)
    return 0.0;
  return GetDataFromDevice(phi_sig, 0);
}

double Beam::GetSigRelativePhi(bool r_good_only) const
{
  if(num_particle < 2)
    return 0.0;
  if(r_good_only)
    return GetDataFromDevice(phi_sig_good, 0);
  else
    return GetDataFromDevice(phi_sig_r, 0);
}

double Beam::GetSigW(bool r_good_only) const
{
  if(num_particle < 2)
    return 0.0;
  if (r_good_only)
    return GetDataFromDevice(w_sig_good, 0);
  else
    return GetDataFromDevice(w_sig, 0);
}

//double Beam::GetSigR() const
//{
//  if(num_particle < 2)
//    return 0.0;
//  return GetDataFromDevice(r_sig, 0);
//}

double Beam::GetEmittanceX() const
{
  if(num_particle < 2)
    return 0.0;
  return GetDataFromDevice(x_emit, 0);
}

double Beam::GetEmittanceY() const
{
  if(num_particle < 2)
    return 0.0;
  return GetDataFromDevice(y_emit, 0);
}

double Beam::GetEmittanceZ() const
{
  if(num_particle < 2)
    return 0.0;
  return GetDataFromDevice(z_emit, 0);
}

uint Beam::GetLossNum() const
{
  return GetDataFromDevice(num_loss, 0);
}

uint Beam::GetLongitudinalLossNum() const
{
  return GetDataFromDevice(num_lloss, 0);
}

uint Beam::GetGoodParticleNum() const
{
  return GetDataFromDevice(num_good, 0);
}

void Beam::ApplyCut(char r_coord, double r_min, double r_max)
{
  char coord = tolower(r_coord);
  double* cd;
  if(coord == 'x')
    cd = x;
  if(coord == 'y')
    cd = y;
  if(coord == 'w')
    cd = w;
  if(coord == 'p')
    cd = phi;
  CutBeamKernelCall(cd, loss, r_min, r_max, num_particle, grid_size, blck_size); 
}

void Beam::ShiftX(double r_val)
{
  return ShiftVariableKernelCall(this, x, r_val);
}

void Beam::ShiftXp(double r_val)
{
  return ShiftVariableKernelCall(this, xp, r_val);
}

void Beam::ShiftY(double r_val)
{
  return ShiftVariableKernelCall(this, y, r_val);
}

void Beam::ShiftYp(double r_val)
{
  return ShiftVariableKernelCall(this, yp, r_val);
}

void Beam::ShiftPhi(double r_val)
{
  ShiftVariableKernelCall(this, phi, r_val);
  InitPhiAvgGood();
}
void Beam::ShiftW(double r_val)
{
  return ShiftVariableKernelCall(this, w, r_val);
}

void Beam::ChangeFrequency(double r_freq)
{
  current *= r_freq/freq;
  std::cout << "Change beam frequency from " << freq << " to " << r_freq << ", ratio = " << r_freq/freq 
    << ", current change to " << current << std::endl;
  ChangeFrequnecyKernelCall(this, r_freq/freq);
  freq = r_freq;
  UpdateAvgPhi();
  UpdateRelativePhi();
  UpdateAvgRelativePhi();
}

