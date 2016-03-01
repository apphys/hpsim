#ifndef SIMULATION_ENGINE_CU_H
#define SIMULATION_ENGINE_CU_H

extern "C"
{
  void SetConstOnDevice(SimulationConstOnDevice*);
  void Init(Beam*, BeamLine*, SpaceCharge*, SimulationParam&);
  void Reset();
  void Cleanup();
  void IncreaseBlIndex();
  void StepSimulationKernelCall(Beam*, BeamLine*, SpaceCharge*, SimulationParam&, uint);
  void SimulateApertureCircular(ApertureCircular*);
  void SimulateApertureRectangular(ApertureRectangular*);
  void SimulateBuncher(Buncher*);
  void SimulateDiagnostics(Diagnostics*);
  void SimulateDipole(Dipole*);
  void SimulateDrift(Drift*);
  void SimulateQuad(Quad*);
  void SimulateDTLRFGap(RFGap*);
  void SimulateCCLRFGap(RFGap*);
  void SimulateRotation(Rotation*);
  void SimulateSpaceChargeCompensation(SpaceChargeCompensation*);
  void SimulateSteerer(Steerer*);
} // extern "C"

#endif
