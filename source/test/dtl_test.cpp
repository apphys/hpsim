#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>
#include "beamline.h"
#include "simulation_engine.h"
#include "beam.h"
#include "init.h"

int main()
{
  BeamLine bl;
  DBConnection dbcon_ptr("../../db/offline-dtl.db");
  GenerateBeamLine(bl, &dbcon_ptr);

  Beam beam("TAEM01_input_beam_64K.dat");
//  beam.InitBeamFromFile("TAEM01_input_beam_64K.dat");

  Scheff scheff(32, 128/2, 3);
  scheff.SetAdjBunchCutoffW(0.8);
  scheff.SetInterval(0.1);
  scheff.SetMeshSizeCutoffW(40.0);

  SimulationEngine se;
  se.InitEngine(&beam, &bl, &scheff);
  se.Simulate();//"01QM00U", "01RG01");
  beam.PrintToFile("end.txt", "");
  //se.Simulate("01QM00U", "02DR02");
  
//  se.Start();
//  std::cout << bl.GetSize() << std::endl;
//  std::cout << bl.GetElementModelIndex("04qm01") << std::endl;
//  std::cout << bl.GetElementName(bl.GetElementModelIndex("04qm01")) << std::endl;
//  std::cout << bl.GetNumOfMonitors() << std::endl;
//  std::vector<uint> indices = bl.GetMonitoredElementsIndices();
//  std::copy(indices.begin(), indices.end(), std::ostream_iterator<uint>(std::cout, ", "));

}
