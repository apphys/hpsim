#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "graphics_common_func.h"
#include "graphics_2d.h"
#include <cuda_gl_interop.h> // after glew.h
#include "beamline.h"
#include "beam.h"
#include "init.h"
#include "init_pv_observer_list.h"
#include "sql_utility.h"
#include "pv_observer_list.h"
#include "server.h"
#include "pthread_check.h"
#include "simulation_engine.h"
#include "plot_data.h"
#include "input_data.h"
#include "epics_put.h"

Beam beam;
BeamLine bl;
SimulationEngine engine;
std::string start_elem;
std::string end_elem;
// data server
PVObserverList uplist;
pthread_t server;
ServerArg* sarg;
pthread_t epicsput;
EPICSPutArg* earg;
// graphics
std::vector<int> glwins;
std::vector<Plot2D> windows;
std::vector<Curve2D*> curves;
PlotData gldata;
std::vector<int> button_states;
std::vector<int> ox, oy;
std::vector<int> subplot_id;
std::vector<float> trans;
const float inertia = 1.5f;
bool pause_simulation = false;
std::vector<std::string> putpvs;

void InitCUDA()
{
  cudaGLSetGLDevice(0); // must appear before cudaGraphicsGLRegisterBuffer and glut init
  cudaDeviceProp devprop;
  cudaGetDeviceProperties(&devprop, 0);
  std::cout << "Continuous mode using " << devprop.name << std::endl; 
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

template<int wid>
void Display()
{
  windows[wid].Display();
}

template<int wid>
void Mouse(int button, int state, int x, int y)
{
  float height = glutGet(GLUT_WINDOW_HEIGHT);
  int num_subplots = windows[wid].plots.size();
  float height_subplot = height/num_subplots;
  subplot_id[wid] = num_subplots - std::ceil(y/height_subplot);
//  if(y < height/num_subplots)
//    subplot_id[wid] = 2;
//  else if(y > 2.0* height/num_subplots)
//    subplot_id[wid] = 0;
//  else
//    subplot_id[wid] = 1;
  if (state == GLUT_DOWN)
      button_states[wid] |= 1<<button;
  else if (state == GLUT_UP)
      button_states[wid] = 0;
  int mods = glutGetModifiers();
  if (mods & GLUT_ACTIVE_CTRL)
      button_states[wid] = 3;
  ox[wid] = x; oy[wid] = y;
  glutPostRedisplay();
}

template<int wid>
void Motion(int x, int y)
{
  float dx, dy;
  dx = (float)(x - ox[wid]);
  dy = (float)(y - oy[wid]);

  if (button_states[wid] == 3) 
  {
    trans[wid] -= (dy / 100.0f) * 0.5f * std::fabs(trans[wid]);
    (windows[wid].plots[subplot_id[wid]])->scale_ratio += (trans[wid] - (windows[wid].plots[subplot_id[wid]])->scale_ratio) * inertia;
  }
  else if (button_states[wid] == 1)
  {
    float height = glutGet(GLUT_WINDOW_HEIGHT)/windows[wid].plots.size();
    float width = glutGet(GLUT_WINDOW_WIDTH);
    if ((windows[wid].plots[subplot_id[wid]])->offset_x > 0)
      (windows[wid].plots[subplot_id[wid]])->offset_mouse_x += dx/width*(windows[wid].plots[subplot_id[wid]])->offset_x*10;
    else
      (windows[wid].plots[subplot_id[wid]])->offset_mouse_x -= dx/width*(windows[wid].plots[subplot_id[wid]])->offset_x*10;
    if ((windows[wid].plots[subplot_id[wid]])->offset_y < 0)
      (windows[wid].plots[subplot_id[wid]])->offset_mouse_y += dy/height*(windows[wid].plots[subplot_id[wid]])->offset_y;
    else
      (windows[wid].plots[subplot_id[wid]])->offset_mouse_y -= dy/height*(windows[wid].plots[subplot_id[wid]])->offset_y;
  }

  ox[wid] = x; oy[wid] = y;
  glutPostRedisplay();
}

template<int wid>
void Key(unsigned char key, int /*x*/, int /*y*/)
{
  switch (key)
  {
    case 'p':
      pause_simulation = !pause_simulation; break;
    case 'q':
      exit(0);
    case 'r':
    {
      (windows[wid].plots[subplot_id[wid]])->ResetScale(); 
      (windows[wid].plots[subplot_id[wid]])->ResetOffSet();
      (windows[wid].plots[subplot_id[wid]])->TurnOnAutoUpdate(); break;
    }
    case 's':
    {
      pause_simulation = true;
      std::cout << "Input x&y range for the " << windows[wid].plots.size() - subplot_id[wid] << " plot: ";
      std::cout << "e.g. 0 100 -5 5" << std::endl;;
      float xmin, xmax, ymin, ymax;
      std::cin >> xmin >> xmax >> ymin >> ymax;
      std::cout << "Set xrange [" << xmin << ", " << xmax << "], yrange [" 
          << ymin << ", " << ymax << "]."<< std::endl;
      (windows[wid].plots[subplot_id[wid]])->TurnOffAutoUpdate();
      (windows[wid].plots[subplot_id[wid]])->ResetScale();
      (windows[wid].plots[subplot_id[wid]])->ResetOffSet();
      (windows[wid].plots[subplot_id[wid]])->SetMaxMin(xmin, xmax, ymin, ymax);
      pause_simulation = false;
      break;
    }
  }
  glutPostRedisplay();
}

template<int wid>
void Menu(int i)
{
  Key<wid>((unsigned char) i, 0, 0);
}

void Idle()
{
  if(!pause_simulation)
  {
    beam.RestoreInitialBeam();
    engine.Simulate(start_elem, end_elem);

    if(!putpvs.empty())
    {
      std::vector<std::vector<double> > vals;
      std::vector<double> loss = gldata.loss_ratio.GetValue();
      std::vector<double> xavg = gldata.xavg.GetValue();
      std::vector<double> xsig = gldata.xsig.GetValue();
      std::vector<double> xpavg = gldata.xpavg.GetValue();
      std::vector<double> xpsig = gldata.xpsig.GetValue();
      std::vector<double> xemit = gldata.xemit.GetValue();
      std::vector<double> yavg = gldata.yavg.GetValue();
      std::vector<double> ysig = gldata.ysig.GetValue();
      std::vector<double> ypavg = gldata.ypavg.GetValue();
      std::vector<double> ypsig = gldata.ypsig.GetValue();
      std::vector<double> yemit = gldata.yemit.GetValue();
      std::vector<double> tmp;
      for(int i = 0; i < putpvs.size(); ++i)
      {
	tmp.assign(16, 0.0);
	tmp[0] = loss[i];
	tmp[1] = xavg[i];
	tmp[2] = xsig[i];
	tmp[3] = xpavg[i];
	tmp[4] = xpsig[i];
	tmp[5] = xemit[i];
	tmp[6] = yavg[i];
	tmp[7] = ysig[i];
	tmp[8] = ypavg[i];
	tmp[9] = ypsig[i];
	tmp[10] = yemit[i];
	vals.push_back(tmp);
      }
      EPICSPut(putpvs, vals);
    }
  }
  for(uint i = 0; i < glwins.size(); ++i)
  {
    glutSetWindow(glwins[i]);
    windows[i].Update();
    glutPostRedisplay();
  }
}

void Cleanup()
{
  //beam.PrintToFile("beam.out");
  std::cout << "------------------------------" << std::endl;
  std::cout << "Cleaning up ..." << std::endl;
  if(!server_stop_flag)
  {
    server_stop_flag = true;
//    epics_put_stop_flag = true;
    void* status;
    ThreadCheck(pthread_join(server, &status), "server join");
    delete sarg;
//    ThreadCheck(pthread_join(epicsput, &status), "epicsput join");
//    delete earg;
  }

  for(int i = 0; i < windows.size(); ++i)
    windows[i].FreePlot();
  for(int i = 0; i < curves.size(); ++i)
    delete curves[i];
//  beam.FreeBeam();
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
  const int win_num = 3;
  const int subplot_num = 4;
  glwins.resize(win_num, 0);
  windows.resize(win_num);
  button_states.resize(win_num);
  ox.resize(win_num);
  oy.resize(win_num);
  subplot_id.resize(win_num);
  trans.assign(win_num, 1.0);

  InitCUDA();
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA|GLUT_ALPHA|GLUT_DOUBLE|GLUT_DEPTH);
  glutInitWindowSize(700, 250*subplot_num);
//---------------------- set up windows
  glwins[0] = glutCreateWindow("Phase Space");
  glewInit();
  windows[0].SetWindowSize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  windows[0].SetBorderWidth(7*10, 2*10);
  windows[0].SetTickSize(10);

//--------------------- setup simulation
//  DBConnection dbcon1("./db/offline-tatd.db");
//  dbcon1.AttachDB("./db/offline-201.db", "dtl");
//  dbcon1.AttachDB("./db/trml.db", "tr");
  DBConnection dbcon1(setting.db[0]);
  for(int i = 1; i < setting.db.size(); ++i)
  {
    std::ostringstream ostr;
    ostr << i;
    dbcon1.AttachDB(setting.db[i], "db" + ostr.str());
  }
//  dbcon1.LoadLib("../../db/lib/libsqliteext.so");
  for(int i = 0; i < setting.libdb.size(); ++i)
    dbcon1.LoadLib(setting.libdb[i]);
  dbcon1.ClearModelIndex();

  GenerateBeamLine(bl, &dbcon1);
  int monitor_num = bl.GetNumOfMonitors(start_elem, end_elem);
  if(monitor_num == 0)
  {
    std::cout << "All monitors are off." << std::endl;
    exit(0);
  }

  const double PI = 3.14159265358979323846;
//  beam.AllocateBeam(32768, 938.272, 1.0, 0.013);
//  beam.InitDCBeam(-0.751, 87.0, 0.000994, -0.398, 113, 0.000493, PI, 0, 0.750, 3);
//  beam.SetRefPhase(-46182.649967/180.0*PI);
//  beam.ShiftPhi(-46182.649967/180.0*PI);
//  beam.SetRefEnergy(0.75);
//  beam.freq = 201.25;
  beam.InitBeamFromFile(setting.beam);
  beam.SaveInitialBeam();
  //beam.PrintToFile("dist_out.dat", "@TADB1");

  gldata.Resize(monitor_num);
  
  Scheff scheff(32, 128/2, 3);
  scheff.SetAdjBunchCutoffW(0.8);
  scheff.SetInterval(0.1);
  scheff.SetMeshSizeCutoffW(40.0);
  engine.InitEngine(&beam, &bl, &scheff, true, &gldata);
  
  engine.Simulate(start_elem, end_elem);

  InitPVObserverList(uplist, bl, dbcon1);

  if (!server_stop_flag)
  {
    sarg = new ServerArg;
  //  sarg->num_monitor = uplist.GetListSize();
    sarg->channels = &uplist;
    ThreadCheck(pthread_create(&server, NULL, ServerRoutine,
                (void*)sarg), "create server thread");
  }

  Histogram2D* h2d1 = new Histogram2D(beam.x, beam.xp, beam.loss, beam.num_particle, 120, 75);
  curves.push_back(h2d1);
  Subplot2D subplot01;
  subplot01.AddCurve(h2d1);
  subplot01.UpdateSubplot();

  Histogram2D* h2d2 = new Histogram2D(beam.y, beam.yp, beam.loss, beam.num_particle, 120, 75);
  curves.push_back(h2d2);
  Subplot2D subplot02;
  subplot02.AddCurve(h2d2);
  subplot02.UpdateSubplot();
   
  Histogram2D* h2d3 = new Histogram2D(beam.phi_r, beam.w, beam.loss, beam.num_particle, 
   beam.phi_avg_r, beam.phi_sig_good, beam.w_avg_good, beam.w_sig_good, 120, 75);
  curves.push_back(h2d3);
  Subplot2D subplot03;
  subplot03.AddCurve(h2d3);
  subplot03.UpdateSubplot();
 

  windows[0].AddSubplot(&subplot03);
  windows[0].AddSubplot(&subplot02);
  windows[0].AddSubplot(&subplot01);

  if(windows[0].InitPlot())
  {
    glutDisplayFunc(Display<0>);
    glutMouseFunc(Mouse<0>);
    glutMotionFunc(Motion<0>);
    glutKeyboardFunc(Key<0>);
    glutCreateMenu(Menu<0>);
    glutAddMenuEntry("[q]uit", 'q');
    glutAddMenuEntry("[r]eset plot", 'r');
    glutAddMenuEntry("[s]et plot range", 's');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
  }
//-------------------- another window --------------------
  glwins[1] = glutCreateWindow("Profiles");
  glewInit();
  windows[1].SetWindowSize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  windows[1].SetBorderWidth(10*7, 10*2);
  windows[1].SetTickSize(10);

  Subplot2D subplot11, subplot12, subplot13, subplot14;
  Histogram* hist1 = new Histogram(beam.x, beam.loss, beam.num_particle, -0.025, 0.025, red);
  hist1->SetLabelY("x profile");
  Histogram* hist2 = new Histogram(beam.y, beam.loss, beam.num_particle, -0.025, 0.025, green);
  hist2->SetLabelY("y profile");
  curves.push_back(hist1);
  curves.push_back(hist2);
  subplot11.AddCurve(hist1);
  subplot11.AddCurve(hist2);
  subplot11.UpdateSubplot();
  Histogram* hist3 = new Histogram(beam.phi_r, beam.loss, beam.num_particle, -M_PI*0.5, M_PI*0.5, yellow);
  hist3->SetLabelY("phase profile");
  curves.push_back(hist3);
  subplot12.SetLabelScaleX(180.0/PI);
  subplot12.AddCurve(hist3);
  subplot12.UpdateSubplot();
  Histogram* hist4 = new Histogram(beam.w, beam.loss, beam.num_particle, 5, 810, blue); // tank4
  hist4->SetLabelY("energy profile");
  curves.push_back(hist4);
  subplot13.AddCurve(hist4);
  subplot13.UpdateSubplot();
  windows[1].AddSubplot(&subplot13);
  windows[1].AddSubplot(&subplot12);
  windows[1].AddSubplot(&subplot11);

  if(windows[1].InitPlot())
  {
    glutDisplayFunc(Display<1>);
    glutMouseFunc(Mouse<1>);
    glutMotionFunc(Motion<1>);
    glutKeyboardFunc(Key<1>);
    glutCreateMenu(Menu<1>);
    glutAddMenuEntry("[q]uit", 'q');
    glutAddMenuEntry("[r]eset plot", 'r');
    glutAddMenuEntry("[s]et plot range", 's');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
  }

//-------------------- another window --------------------
  glwins[2] = glutCreateWindow("Sigmas & Emittances");
  glewInit();
  windows[2].SetWindowSize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  windows[2].SetBorderWidth(10*7, 10*2);
  windows[2].SetTickSize(10);

  PlotDataMember<double> xaxis(monitor_num);  
  double* h_axis = new double[monitor_num];
  std::vector<uint> mi = bl.GetMonitoredElementsIndices(start_elem, end_elem);
  std::vector<double> md(mi.begin(), mi.end());
  std::copy(md.begin(), md.end(), h_axis);
  cudaMemcpy(xaxis.d_ptr, h_axis, sizeof(double)*monitor_num, cudaMemcpyHostToDevice);
  delete [] h_axis;

  Curve2D* cxsig = new Curve2D(xaxis.d_ptr, gldata.xsig.d_ptr, monitor_num, red);
  cxsig->SetLabelY("x sigma");
  Curve2D* cysig = new Curve2D(xaxis.d_ptr, gldata.ysig.d_ptr, monitor_num, green);
  cysig->SetLabelY("y sigma");
  curves.push_back(cxsig);
  curves.push_back(cysig);
  Subplot2D subplot21;
  subplot21.AddCurve(cxsig);
  subplot21.AddCurve(cysig);
  subplot21.UpdateSubplot();
  Curve2D* cphisig = new Curve2D(xaxis.d_ptr, gldata.phisig.d_ptr, monitor_num, yellow);
  cphisig->SetLabelY("phase sigma");
  curves.push_back(cphisig);
  Subplot2D subplot22;
  subplot22.SetTickLabelScaleY(180.0/PI);
  subplot22.AddCurve(cphisig);
  subplot22.UpdateSubplot();
  Curve2D* cwsig = new Curve2D(xaxis.d_ptr, gldata.wsig.d_ptr, monitor_num, blue);
  cwsig->SetLabelY("energy sigma");
  curves.push_back(cwsig);
  Subplot2D subplot23;
  subplot23.AddCurve(cwsig);
  subplot23.UpdateSubplot();
  Curve2D* cemitx = new Curve2D(xaxis.d_ptr, gldata.xsig.d_ptr, monitor_num, red);
  cemitx->SetLabelY("x emittance");
  Curve2D* cemity = new Curve2D(xaxis.d_ptr, gldata.yemit.d_ptr, monitor_num, green);
  cemity->SetLabelY("y emittance");
  curves.push_back(cemitx);
  curves.push_back(cemity);
  Subplot2D subplot24;
  subplot24.AddCurve(cemitx);
  subplot24.AddCurve(cemity);
  subplot24.UpdateSubplot();
  //Curve2D* cemitz = new Curve2D(xaxis.d_ptr, gldata.zemit.d_ptr, monitor_num, cyan);
  //cemitz->SetLabelY("z emittance");
  //curves.push_back(cemitz);
  //Subplot2D subplot25;
  //subplot25.AddCurve(cemitz);
  //subplot25.UpdateSubplot();
  Curve2D* closs_loc = new Curve2D(xaxis.d_ptr, gldata.loss_local.d_ptr, monitor_num, orange);
  closs_loc->SetLabelY("local loss");
  curves.push_back(closs_loc);
  Subplot2D subplot25;
  subplot25.AddCurve(closs_loc);
  subplot25.UpdateSubplot();

  Curve2D* closs = new Curve2D(xaxis.d_ptr, gldata.loss_ratio.d_ptr, monitor_num, magenta);
  closs->SetLabelY("cumulative loss");
  curves.push_back(closs);
  Subplot2D subplot26;
  subplot26.AddCurve(closs);
  subplot26.UpdateSubplot();

  windows[2].AddSubplot(&subplot26);
  windows[2].AddSubplot(&subplot25);
  windows[2].AddSubplot(&subplot24);
  windows[2].AddSubplot(&subplot23);
  windows[2].AddSubplot(&subplot22);
  windows[2].AddSubplot(&subplot21);

  if(windows[2].InitPlot())
  {
    glutDisplayFunc(Display<2>); 
    glutMouseFunc(Mouse<2>);
    glutMotionFunc(Motion<2>);
    glutKeyboardFunc(Key<2>);
    glutCreateMenu(Menu<2>);
    glutAddMenuEntry("[q]uit", 'q');
    glutAddMenuEntry("[r]eset plot", 'r');
    glutAddMenuEntry("[s]et plot range", 's');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
  }

//-------------------- another window --------------------
//  glwins[3] = glutCreateWindow("Averages & Loss");
//  glewInit();
//  windows[3].SetWindowSize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
//  windows[3].SetBorderWidth(10*7, 10*2);
//  windows[3].SetTickSize(10);
//
//  Curve2D* cavgx = new Curve2D(xaxis.d_ptr, gldata.xavg.d_ptr, monitor_num, red);
//  cavgx->SetLabelY("x mean");
//  Curve2D* cavgy = new Curve2D(xaxis.d_ptr, gldata.yavg.d_ptr, monitor_num, green);
//  cavgy->SetLabelY("y mean");
//  curves.push_back(cavgx);
//  curves.push_back(cavgy);
//  Subplot2D subplot31;
//  subplot31.AddCurve(cavgx);
//  subplot31.AddCurve(cavgy);
//  subplot31.UpdateSubplot();
//  Curve2D* cavgphi = new Curve2D(xaxis.d_ptr, gldata.phiavg.d_ptr, monitor_num, yellow);
//  cavgphi->SetLabelY("realtive phase mean");
//  curves.push_back(cavgphi);
//  Subplot2D subplot32;
//  subplot32.SetTickLabelScaleY(180/PI);
//  subplot32.AddCurve(cavgphi);
//  subplot32.UpdateSubplot();
//  Curve2D* cavgw = new Curve2D(xaxis.d_ptr, gldata.wavg.d_ptr, monitor_num, blue);
//  cavgw->SetLabelY("relative energy mean");
//  curves.push_back(cavgw);
//  Subplot2D subplot33;
//  subplot33.AddCurve(cavgw);
//  subplot33.UpdateSubplot();
//  Curve2D* closs = new Curve2D(xaxis.d_ptr, gldata.loss_ratio.d_ptr, monitor_num, magenta);
//  closs->SetLabelY("cumulative loss");
//  Curve2D* closs_loc = new Curve2D(xaxis.d_ptr, gldata.loss_local.d_ptr, monitor_num, orange);
//  closs_loc->SetLabelY("local loss");
//  curves.push_back(closs);
//  curves.push_back(closs_loc);
//  Subplot2D subplot34;
//  subplot34.AddCurve(closs);
//  subplot34.AddCurve(closs_loc);
//  subplot34.UpdateSubplot();
//
//  windows[3].AddSubplot(&subplot34);
//  windows[3].AddSubplot(&subplot33);
//  windows[3].AddSubplot(&subplot32);
//  windows[3].AddSubplot(&subplot31);
//
//  if(windows[3].InitPlot())
//  {
//    glutDisplayFunc(Display<3>); 
//    glutMouseFunc(Mouse<3>);
//    glutMotionFunc(Motion<3>);
//    glutKeyboardFunc(Key<3>);
//    glutCreateMenu(Menu<3>);
//    glutAddMenuEntry("[q]uit", 'q');
//    glutAddMenuEntry("[r]eset plot", 'r');
//    glutAddMenuEntry("[s]et plot range", 's');
//    glutAttachMenu(GLUT_RIGHT_BUTTON);
//  }
//------------------------------------
  glutIdleFunc(Idle);
  atexit(Cleanup);

  glutMainLoop();
}
