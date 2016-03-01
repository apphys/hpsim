#ifndef GRAPHICS_3D_H        
#define GRAPHICS_3D_H        

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "graphics_common_func.h"
#include "plot_data_3d.h"

class DotPlot3D
{
public:
  void Init(PlotData3D*, uint);
  void Free();
  void Display();
  void Update();
  void ResetColor();

  uint data_num;
  float point_size;

private:
  void InitGL();
  void DrawPoints();
  float4* data_; 
  GLuint pos_vbo_;
  GLuint color_vbo_;
  struct cudaGraphicsResource* cuda_pos_vbo_;
  GLuint program_;
  PlotData3D* input_;
  uint blck_num_;
  uint thrd_num_;
};

#endif
