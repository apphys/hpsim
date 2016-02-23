#include <iostream>
#include "beam.h"

int main()
{
  Beam beam("Hm_clz_64k.txt");
  uint lossnum1 = beam.GetLossNum();
  std::cout << "Beam loss num = " << lossnum1 << std::endl;
  beam.UpdateLoss();
  uint lossnum2 = beam.GetLossNum();
  std::cout << "Beam loss num = " << lossnum2 << std::endl;
}
