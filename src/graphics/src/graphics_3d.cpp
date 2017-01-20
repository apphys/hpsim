#include <cmath>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <unistd.h>
#include "glew.h"            // must precede cuda_gl_interop.h
#include <cuda_gl_interop.h> // must be after glew.h
#include "graphics_common_func.h"
#include "graphics_3d.h"
#include "graphics_kernel_call_cu.h"

void DotPlot3D::Init(PlotData3D* r_input, uint r_data_num)
{
  data_num = r_data_num;
  point_size = 3.0f;
  program_ = 0;
  pos_vbo_ = 0;
  color_vbo_ = 0;
  InitGL();

  input_ = r_input;
//  data_ = new float4[r_data_num];
  ResetColor();
  unsigned long total_thread_num = std::pow(2, (int)std::ceil(std::log(r_data_num)/std::log(2)));
  if(total_thread_num > 512)
  {
    blck_num_ = total_thread_num/512;
    thrd_num_ = 512;
  }
  else
  {
    blck_num_ = 1;
    thrd_num_ = total_thread_num;
  }
}

void DotPlot3D::Free()
{
  if(pos_vbo_)
  {
    cudaGraphicsUnregisterResource(cuda_pos_vbo_);
    glDeleteBuffers(1, (const GLuint*)&pos_vbo_);
  }
  if(color_vbo_)
    glDeleteBuffers(1, (const GLuint*)&color_vbo_);
//  if(data_)
//    delete [] data_;
}

void DotPlot3D::InitGL()
{
  GLuint vshader, fshader;
  if ((fshader = CreateShader(std::string(GetProjectTopDir() + "/src/graphics/src/graphics_3d.f.glsl").c_str(), GL_FRAGMENT_SHADER)) == 0) exit(-1);
  if ((vshader = CreateShader(std::string(GetProjectTopDir() + "/src/graphics/src/graphics_3d.v.glsl").c_str(), GL_VERTEX_SHADER))   == 0) exit(-1);

  GLuint program = glCreateProgram();
  glAttachShader(program, vshader);
  glAttachShader(program, fshader);
  glLinkProgram(program);
  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if(!success)
  {
    char temp[256];
    glGetProgramInfoLog(program, 256, 0, temp);
    printf("Failed to link program:\n%s\n", temp);
    glDeleteProgram(program);
    program = 0;
  }
  program_ = program;
  pos_vbo_ = CreateVBO(data_num*sizeof(float4));
  cudaGraphicsGLRegisterBuffer(&cuda_pos_vbo_, pos_vbo_, cudaGraphicsMapFlagsNone);
  color_vbo_ = CreateVBO(data_num*sizeof(float4));
}

void DotPlot3D::DrawPoints()
{
  glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_);
  glVertexPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
  glColorPointer(4, GL_FLOAT, 0, 0);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, data_num);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
}

void DotPlot3D::Display()
{
  glColor3f(1, 1, 1);
  glPointSize(point_size);
  DrawPoints();
}

void DotPlot3D::ResetColor()
{
  std::vector<double> z = input_->GetPhi();
  std::vector<uint> loss = input_->GetLoss();
  glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);  
  float* color_data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
  for(int i = 0; i < data_num; ++i)
  {
    if(loss[i] == 0)
    {
      double maxval = M_PI;
      ColorRamp((z[i] + maxval)/maxval *0.5, color_data);
      color_data += 3;  
      *color_data = 1.0f; 
      ++color_data;
    }
    else
    {
      color_data[0] = 0.0;
      color_data[1] = 0.0;
      color_data[2] = 0.0;
      color_data[3] = 0.0;
      color_data += 4;
    }
  }
  glUnmapBuffer(GL_ARRAY_BUFFER);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void DotPlot3D::Update()
{
  cudaGraphicsMapResources(1, &cuda_pos_vbo_, 0); // count, resource, stream
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&data_, &num_bytes, cuda_pos_vbo_);
  Set3dData(blck_num_, thrd_num_, input_->x, input_->y, input_->phi, data_, data_num);
  cudaGraphicsUnmapResources(1, &cuda_pos_vbo_, 0);
  
  std::vector<uint> loss = input_->GetLoss();
  glBindBuffer(GL_ARRAY_BUFFER, color_vbo_);
  float4* color_data = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
  for(int i = 0; i < data_num; ++i)
  {
    if(loss[i] != 0)
    {
      color_data[i].x = 0.0;
      color_data[i].y = 0.0;
      color_data[i].z = 0.0;
      color_data[i].w = 0.0;
    }
  }
  glUnmapBuffer(GL_ARRAY_BUFFER);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}
