#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "beamline.h"
#include "init.h"
#include "sql_utility.h"
#include "server.h"
#include "pthread_check.h"
#include "simulation_engine.h"
#include "plot_data.h"
#include "input_data.h"
#include "epics_put.h"
#include "init_pv_observer_list.h"
#include "pv_observer_list.h"

#include <typeinfo>

BeamLine bl;
SimulationEngine engine;
std::string start_elem;
std::string end_elem;
// data server
PVObserverList oblist;
pthread_t server;
ServerArg* sarg;
pthread_t epicsput;
EPICSPutArg* earg;
std::vector<std::string> putpvs;

void InitCUDA()
{
//  cudaGLSetGLDevice(2); // must appear before cudaGraphicsGLRegisterBuffer and glut init
  cudaSetDevice(2);
  cudaDeviceProp devprop;
  cudaGetDeviceProperties(&devprop, 0);
  std::cout << "Continuous mode using " << devprop.name << std::endl; 
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

void Cleanup()
{
  std::cout << "------------------------------" << std::endl;
  std::cout << "Cleaning up ..." << std::endl;
  void* status;
  ThreadCheck(pthread_join(server, &status), "server join");
  delete sarg;
}

void* WaitInput(void* arg)
{
  char input;
  while(std::cin >> input)
  {
    if (input == 'q' || input == 'Q')
    {
      std::cout << "Quit ..." << std::endl;
      server_stop_flag = true;
      break;
    }
  }
  pthread_exit(NULL);
}

void LoopSimulation()
{
  while(!server_stop_flag)
  {
    std::vector<std::vector<double> > vals;
    std::vector<double> tmp;
    tmp.assign(16, 0.0);
//    Dipole* d1 = dynamic_cast<Dipole*>(bl[4]);
//    tmp[0] = d1->GetRadius();
//    tmp[1] = d1->GetAngle();
//    tmp[2] = d1->GetEdgeAngleIn();
//    tmp[3] = d1->GetEdgeAngleOut();
//    tmp[4] = d1->GetKineticEnergy();
//    ////////////////////////////////
//    Buncher* b1 = dynamic_cast<Buncher*>(bl[42]);
//    Buncher* b2 = dynamic_cast<Buncher*>(bl[87]);
//    tmp[2] = b1->GetVoltage();
//    tmp[3] = b1->GetPhase();
//    tmp[4] = (b1->IsOn() ? 1 : 0);
//    tmp[5] = b2->GetVoltage();
//    tmp[6] = b2->GetPhase();
//    tmp[7] = (b2->IsOn() ? 1 : 0);
//    /////////////////////////////
//    Quad* qd = dynamic_cast<Quad*>(bl[0]);
//    tmp[0] = (qd->GetGradient());
//    qd = dynamic_cast<Quad*>(bl[1]);
//    tmp[1] = (qd->GetGradient());
//    RFGap* gp = dynamic_cast<RFGap*>(bl[2]);
//    tmp[2] = gp->GetRFAmplitude();
//    tmp[3] = gp->GetRefPhase();
//    tmp[4] = gp->GetPhaseShift();
//    gp = dynamic_cast<RFGap*>(bl[141]);
//    tmp[5] = gp->GetRFAmplitude();
//    tmp[6] = gp->GetRefPhase();
//    tmp[7] = gp->GetPhaseShift();
//    gp = dynamic_cast<RFGap*>(bl[251]);
//    tmp[8] = gp->GetRFAmplitude();
//    tmp[9] = gp->GetRefPhase();
//    tmp[10] = gp->GetPhaseShift();
//    gp = dynamic_cast<RFGap*>(bl[339]);
//    tmp[11] = gp->GetRFAmplitude();
//    tmp[12] = gp->GetRefPhase();
//    tmp[13] = gp->GetPhaseShift();

    vals.push_back(tmp);
    EPICSPut(putpvs, vals);
  }
}

int main(int argc, char* argv[])
{
  std::string setting_file;
  if(argc > 1)
  {
    setting_file = std::string(argv[1]);
  }
  else
  {
    std::cerr << "Must provide a input file." << std::endl;
    exit(-1);
  }
  InputData setting = ProcessInputFile(setting_file);
  server_stop_flag = setting.server_off;
  if(!setting.put_pvs.empty())
    for(int i = 0; i < setting.put_pvs.size(); ++i)
      putpvs.push_back(setting.put_pvs[i]);
  start_elem = setting.start;
  end_elem = setting.end;

  InitCUDA();
//--------------------- setup simulation
  DBConnection dbcon1(setting.db[0]);
  for(int i = 1; i < setting.db.size(); ++i)
  {
    std::ostringstream ostr;
    ostr << i;
    dbcon1.AttachDB(setting.db[i], "db" + ostr.str());
  }
  dbcon1.LoadLib("./lib/libsqliteext.so");
  dbcon1.ClearModelIndex();

  GenerateBeamLine(bl, &dbcon1);

//----------------------- start data server
  InitPVObserverList(oblist, bl, dbcon1);
  oblist.Print();

  if (!server_stop_flag)
  {
    sarg = new ServerArg;
    sarg->channels = &oblist;
    ThreadCheck(pthread_create(&server, NULL, ServerRoutine,
                (void*)sarg), "create server thread");
  }

  pthread_t inputthread;
  ThreadCheck(pthread_create(&inputthread, NULL, WaitInput, NULL), "create inputthread");
  LoopSimulation();
  void* status;
  ThreadCheck(pthread_join(inputthread, &status), "inputthread join");
  Cleanup();
}
