#include "plot_data.h"

PlotData::PlotData()
{
}

PlotData::PlotData(int r_sz) : data_size(r_sz),
  xavg(r_sz), xsig(r_sz), xemit(r_sz), 
  yavg(r_sz), ysig(r_sz), yemit(r_sz), 
  phiavg(r_sz), phisig(r_sz), wavg(r_sz), 
  wsig(r_sz), zemit(r_sz), loss_ratio(r_sz),
  loss_local(r_sz)
{
}

void PlotData::Resize(int r_sz)
{
  xavg.Resize(r_sz);
  xsig.Resize(r_sz);
  xpavg.Resize(r_sz);
  xpsig.Resize(r_sz);
  xemit.Resize(r_sz);
  yavg.Resize(r_sz);
  ysig.Resize(r_sz);
  ypavg.Resize(r_sz);
  ypsig.Resize(r_sz);
  xemit.Resize(r_sz);
  yemit.Resize(r_sz);
  phiavg.Resize(r_sz);
  phisig.Resize(r_sz);
  wavg.Resize(r_sz);
  wsig.Resize(r_sz);
  zemit.Resize(r_sz);
  loss_ratio.Resize(r_sz);
  loss_local.Resize(r_sz);
}
