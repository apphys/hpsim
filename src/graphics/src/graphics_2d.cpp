#include "graphics_2d.h" // this has to be included before <cuda_gl_interop.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <GL/glm/glm.hpp>
#include <GL/glm/gtc/matrix_transform.hpp>
#include <GL/glm/gtc/type_ptr.hpp>
#include "graphics_common_func.h"
#include "graphics_kernel_call_cu.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

Curve2D::Curve2D()
{
}

Curve2D::Curve2D(double* r_input_x, double* r_input_y, uint r_data_num, 
                 GLfloat* r_color, bool r_auto_update, bool r_data_tracker_on) 
{
  Initialize(r_input_x, r_input_y, r_data_num, r_color, r_auto_update, r_data_tracker_on);
}

void Curve2D::Initialize(double* r_input_x, double* r_input_y, uint r_data_num, 
                 GLfloat* r_color, bool r_auto_update, bool r_data_tracker_on)
{

  input_x = r_input_x; input_y = r_input_y; data_num = r_data_num; 
  color = r_color; auto_update = r_auto_update; 
  data_tracker_on = r_data_tracker_on;
  label_x = ""; label_y = "";
  // create vbo
  glGenBuffers(1, &vbo); 
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float2)*r_data_num, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  // register the vbo with cuda resource
  cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, cudaGraphicsMapFlagsNone);
  
  maxmin = new double4;
  // decide grid size(blck_num) and block size(thrd_num)
  unsigned long total_thread_num = std::pow(2, (int)std::ceil(std::log(r_data_num)/std::log(2))); 
  if(total_thread_num > 512) 
  {
    blck_num = total_thread_num/512;
    thrd_num = 512;
  }
  else
  {
    blck_num = 1;
    thrd_num = total_thread_num;
  }

  // Allocate mem for partial results for max & min
  cudaMalloc((void**)&partial_x2, sizeof(double)*(blck_num+1)*2); // replace 2 with sz of blocks+1
  cudaMalloc((void**)&partial_y2, sizeof(double)*(blck_num+1)*2); // replace 2 with sz of blocks+1
}

Curve2D::~Curve2D()
{
  cudaGraphicsUnregisterResource(cuda_vbo);
  glDeleteBuffers(1, (const GLuint*)&vbo);
  delete maxmin;
  cudaFree(partial_x2);
  cudaFree(partial_y2);
  std::cout << "Curve2D is freed." << std::endl;
}

void Curve2D::UpdateData()
{
  cudaGraphicsMapResources(1, &cuda_vbo, 0); // count, resource, stream
  size_t num_bytes; 
  cudaGraphicsResourceGetMappedPointer((void**)&data, &num_bytes, cuda_vbo);

  // call kernel to rearrange data and copy from device to device
  Set2dCurveData(blck_num, thrd_num, input_x, input_y, data, data_num, &last_data);
  if(auto_update)
    FindMaxMin2D(blck_num, thrd_num, input_x, input_y, partial_x2, partial_y2, maxmin, data_num);
  cudaGraphicsUnmapResources(1, &cuda_vbo, 0);

#ifdef _DEBUG
  std::cout << "curve x (" << GetMinX() << ", " << GetMaxX() << ") "
            << ", y (" << GetMinY() << ", " << GetMaxY() << ") " << std::endl;
#endif 
}

/*------------- Histogram2D ----------------*/
Histogram2D::Histogram2D(double* r_input_x, double* r_input_y, uint* r_input_loss,
  uint r_data_num, uint r_bin_num_x, uint r_bin_num_y)
  : Curve2D(r_input_x, r_input_y, r_bin_num_x*r_bin_num_y, NULL, true, false), input_loss(r_input_loss),
    bin_num_x(r_bin_num_x), bin_num_y(r_bin_num_y), raw_data_num(r_data_num)
{
  InitHistogram2D();
  auto_update = true;
}

Histogram2D::Histogram2D(double* r_input_x, double* r_input_y, uint* r_input_loss,
  uint r_data_num, double* r_avg_x, double* r_sig_x, double* r_avg_y, double* r_sig_y,
  uint r_bin_num_x, uint r_bin_num_y)
  : Curve2D(r_input_x, r_input_y, r_bin_num_x*r_bin_num_y, NULL, false, false), input_loss(r_input_loss),
    bin_num_x(r_bin_num_x), bin_num_y(r_bin_num_y), raw_data_num(r_data_num)
{
  InitHistogram2D();
  auto_update = false;
  avg_x = r_avg_x;
  avg_y = r_avg_y;
  sig_x = r_sig_x;
  sig_y = r_sig_y;
}

void Histogram2D::InitHistogram2D()
{
  // create vbo
  glGenBuffers(1, &color_vbo); 
  glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float4)*data_num, 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  // register the vbo with cuda resource
 // cudaGraphicsGLRegisterBuffer(&cuda_color_vbo, color_vbo, cudaGraphicsMapFlagsNone);

  SetMaxMin(std::numeric_limits<double>::min(), std::numeric_limits<double>::max(), 
    std::numeric_limits<double>::min(), std::numeric_limits<double>::max());

  thrd_num = 128;
  uint total_thread_num = std::pow(2, (int)std::ceil(std::log(raw_data_num)/std::log(2)));
  blck_num = total_thread_num / thrd_num;

  cudaMalloc((void**)&hist, sizeof(uint)*data_num);
  cudaMemset(hist, 0, sizeof(uint)*data_num);
}

Histogram2D::~Histogram2D()
{
//  cudaGraphicsUnregisterResource(cuda_color_vbo);
  glDeleteBuffers(1, (const GLuint*)&color_vbo);
}

void Histogram2D::UpdateData()
{
  if (auto_update)
  {
//    FindMaxMin2D(blck_num, thrd_num, input_x, input_y, partial_x2, partial_y2, maxmin, raw_data_num, input_loss);
//    std::cout << "@@@@Histogram2D, maxmin: " << maxmin->y << ", " << maxmin->x << ", " << maxmin->w << ", " << maxmin->z << std::endl; 
    double* x_h = new double[raw_data_num];
    cudaMemcpy(x_h, input_x, sizeof(double)*raw_data_num, cudaMemcpyDeviceToHost);
    double* y_h = new double[raw_data_num];
    cudaMemcpy(y_h, input_y, sizeof(double)*raw_data_num, cudaMemcpyDeviceToHost);
    uint* loss_h = new uint[raw_data_num];
    cudaMemcpy(loss_h, input_loss, sizeof(uint)*raw_data_num, cudaMemcpyDeviceToHost);
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::min();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::min();
    for (uint i = 0; i < raw_data_num; ++i)
    {
      if (loss_h[i] == 0) 
      {
        if (x_h[i] < xmin) xmin = x_h[i];
        if (x_h[i] > xmax) xmax = x_h[i];
        if (y_h[i] < ymin) ymin = y_h[i];
        if (y_h[i] > ymax) ymax = y_h[i];
      }
    }
    maxmin->y = xmin;//*std::min_element(x_h, x_h+raw_data_num);
    maxmin->x = xmax;//*std::max_element(x_h, x_h+raw_data_num);
    maxmin->w = ymin;//*std::min_element(y_h, y_h+raw_data_num);
    maxmin->z = ymax;//*std::max_element(y_h, y_h+raw_data_num);

//    std::cout << "Real beam maxmin: " << *std::min_element(x_h, x_h+raw_data_num) << ", " << *std::max_element(x_h, x_h+raw_data_num)
//      <<", " << *std::min_element(y_h, y_h+raw_data_num) << ", " << *std::max_element(y_h, y_h+raw_data_num) << std::endl;
    delete [] x_h;
    delete [] y_h;
    delete [] loss_h;
  }
  else
  {
    double avg_x_h, avg_y_h, sig_x_h, sig_y_h;
    cudaMemcpy(&avg_x_h, avg_x, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&avg_y_h, avg_y, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sig_x_h, sig_x, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sig_y_h, sig_y, sizeof(double), cudaMemcpyDeviceToHost);
    SetMaxMin(avg_x_h-6*sig_x_h, avg_x_h+6*sig_x_h, avg_y_h-6*sig_y_h, avg_y_h+6*sig_y_h);
  }

  cudaGraphicsMapResources(1, &cuda_vbo, 0); // count, resource, stream
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&data, &num_bytes, cuda_vbo);
  SetHistogram2DCoordinateDataKernelCall(data, maxmin, bin_num_x, bin_num_y, thrd_num, blck_num); 
  cudaGraphicsUnmapResources(1, &cuda_vbo, 0);

  cudaMemset(hist, 0, sizeof(uint)*data_num);
  UpdateHistogram2DKernelCall(hist, input_x, input_y, input_loss, raw_data_num, maxmin, 
    bin_num_x, bin_num_y, thrd_num, blck_num);
  uint* hist_h = new uint[data_num];
  cudaMemcpy(hist_h, hist, sizeof(uint)*data_num, cudaMemcpyDeviceToHost);
  uint maxval = *std::max_element(hist_h, hist_h + data_num);  

  glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
//  float* color_data = new float[data_num*4];
  float* color_data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
  for (uint i = 0; i < data_num; ++i)
  {
    if(maxval < 1)
    {
      color_data[0] = 1.f;
      color_data[1] = 1.f;
      color_data[2] = 1.f;
    }
    else
    {
      double r = (double) hist_h[i] / maxval;
      ColorRampWhiteBackGround(r, color_data, maxval);
    }
    color_data += 3;
    *color_data = 1.0f;
    ++color_data;
  }
//  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*data_num*4, color_data, GL_DYNAMIC_DRAW);
  glUnmapBuffer(GL_ARRAY_BUFFER);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  delete [] hist_h;
}

/*------------- PhaseSpace ----------------*/
PhaseSpace::PhaseSpace(double* r_input_x, double* r_input_y, uint* r_input_loss,
  uint r_data_num, GLfloat* r_color, bool r_auto_update) 
  : Curve2D(r_input_x, r_input_y, r_data_num, r_color, r_auto_update, false), 
    input_loss(r_input_loss)
{
  cudaMalloc((void**)&tmp_x, sizeof(double)*r_data_num);
  cudaMalloc((void**)&tmp_y, sizeof(double)*r_data_num);
  cudaMalloc((void**)&tmp_loss, sizeof(uint)*r_data_num);
}

void PhaseSpace::UpdateData()
{
  cudaGraphicsMapResources(1, &cuda_vbo, 0); // count, resource, stream
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&data, &num_bytes, cuda_vbo);
  SetPhaseSpaceData(blck_num, thrd_num, input_x, input_y, input_loss,
                    tmp_x, tmp_y, tmp_loss, data, data_num);
  if(auto_update)
    FindMaxMin2D(blck_num, thrd_num, tmp_x, tmp_y, partial_x2, partial_y2, maxmin, data_num);
  cudaGraphicsUnmapResources(1, &cuda_vbo, 0);
}

PhaseSpace::~PhaseSpace()
{
  cudaFree(tmp_x);
  cudaFree(tmp_y);
  cudaFree(tmp_loss);
  std::cout << "PhaseSpace is Freed. " << std::endl;
}

/*------------- Histogram -----------------*/
Histogram::Histogram(double* r_input, uint* r_input_loss, uint r_data_num, 
               double r_min, double r_max, GLfloat* r_color, uint r_bin_num)
  : Curve2D(NULL, r_input, r_bin_num, r_color, false, false), max(r_max), 
    min(r_min), input_loss(r_input_loss), hist_data_num(r_data_num)
{
  cudaMalloc((void**)&input_x, sizeof(double)*r_bin_num);
  double width = (r_max-r_min)/r_bin_num;
  double* h_input_x = new double[r_bin_num];
  for(int i = 0; i < r_bin_num; ++i)
    h_input_x[i] = r_min + width*(0.5 + i); 
  cudaMemcpy(input_x, h_input_x, sizeof(double)*r_bin_num, cudaMemcpyHostToDevice);
  delete h_input_x;

  thrd_num = 128;
  uint total_thread_num = std::pow(2, (int)std::ceil(std::log(hist_data_num)/std::log(2)));
  blck_num = total_thread_num / thrd_num;

  cudaMalloc((void**)&hist, sizeof(uint)*r_bin_num);
  cudaMemset(hist, 0, sizeof(uint)*r_bin_num);
  cudaMalloc((void**)&partial_hist, sizeof(uint)*blck_num*r_bin_num);
  cudaMemset(partial_hist, 0, sizeof(uint)*blck_num*r_bin_num);
  SetMaxMin(r_min, r_max, -1, std::log(r_data_num)/std::log(2.0));
}

Histogram::~Histogram()
{
  cudaFree(input_x);
  cudaFree(hist);
  cudaFree(partial_hist);
  std::cout << "Histogram is Freed. " << std::endl;
}

void Histogram::UpdateData()
{
  cudaGraphicsMapResources(1, &cuda_vbo, 0); // count, resource, stream
  size_t num_bytes;
  cudaGraphicsResourceGetMappedPointer((void**)&data, &num_bytes, cuda_vbo);
//  FindMaxMin1D(blck_num, thrd_num, input_y, input_loss, partial_y2, maxmin, hist_data_num);
//  max = maxmin->y;
//  min = maxmin->x;
  UpdateHistogram(input_y, input_loss, hist, partial_hist, hist_data_num, data_num, min, max, blck_num, thrd_num); 
//  std::cout << "Histogram data_num = " << data_num << ", thrd_num = " << thrd_num << ", blck_num = " << blck_num << std::endl;
  double width = (max-min)/data_num;
  double* h_input_x = new double[data_num];
  for(int i = 0; i < data_num; ++i)
    h_input_x[i] = min + width*(0.5 + i); 
  cudaMemcpy(input_x, h_input_x, sizeof(double)*data_num, cudaMemcpyHostToDevice);
  delete h_input_x;

  Set2dHistogramData(blck_num, thrd_num, input_x, hist, data, data_num); // data_num = bin_num
  cudaGraphicsUnmapResources(1, &cuda_vbo, 0);
}

/*------------- Subplot2D -----------------*/
Subplot2D::Subplot2D() 
  : offset_x(0.0), offset_y(0.0), scale_x(1.0), scale_y(1.0), 
    offset_x_init(0.0), offset_y_init(0.0), scale_x_init(1.0), 
    scale_y_init(1.0), scale_ratio(1.0), show_grids(false),
    offset_mouse_x(0.0), offset_mouse_y(0.0), label_scale_x(1.0),
    label_scale_y(1.0)
{
  xmarker.resize(0);
  ymarker.resize(0);
}

void Subplot2D::AddCurve(Curve2D* r_curve)
{
  curves.push_back(r_curve);
}
void Subplot2D::UpdateSubplot()
{
  if(curves.empty()) return;
  double xmax, xmin, ymax, ymin;
  std::vector<double> vxmax(curves.size(), 0.0);
  std::vector<double> vxmin(curves.size(), 0.0);
  std::vector<double> vymax(curves.size(), 0.0);
  std::vector<double> vymin(curves.size(), 0.0);
  for(uint i = 0; i < curves.size(); ++i)
  {
    curves[i]->UpdateData();
    vxmax[i] = curves[i]->GetMaxX();
    vxmin[i] = curves[i]->GetMinX();
    vymax[i] = curves[i]->GetMaxY();
    vymin[i] = curves[i]->GetMinY();
#ifdef _DEBUG
    std::cout << "Curve " << i << ": xmax=" << vxmax[i] << ", xmin=" << vxmin[i] 
              << ", ymax=" << vymax[i] << ", ymin=" << vymin[i] << std::endl;
#endif
  }
  xmax = *std::max_element(vxmax.begin(), vxmax.end());
  xmin = *std::min_element(vxmin.begin(), vxmin.end());
  ymax = *std::max_element(vymax.begin(), vymax.end());
  ymin = *std::min_element(vymin.begin(), vymin.end());
#ifdef _DEBUG
  std::cout << "For this Supblot, xmax= " << xmax << ", xmin=" << xmin 
    << ", ymax=" << ymax << ", ymin=" << ymin << std::endl;
#endif
  scale_x = scale_x_init = 2.0/(xmax-xmin)*0.95;
  scale_x *= scale_ratio;
  scale_y = scale_y_init = 2.0/(ymax-ymin)*0.85;
  scale_y *= scale_ratio;
  offset_x = offset_x_init = -(xmin+xmax)/2.0;
  offset_x += offset_mouse_x;
  offset_y = offset_y_init = -(ymin+ymax)/2.0;
  offset_y += offset_mouse_y;
}
/*----------------- Plot2D --------------------*/
//Plot2D::Plot2D()
//{
//}
//Plot2D::~Plot2D()
//{
//  FreePlot();
//}
int Plot2D::InitPlot()
{
  // init 2D shaders for uniform color
  GLint link_ok = GL_FALSE;
  GLuint vs, fs;
  /*!
   * /todo fix CreateShader graph_2d.v.glsl file path
   */
  if((vs = CreateShader(std::string(GetProjectTopDir() + "/src/graphics/src/graph_2d.v.glsl").c_str(), GL_VERTEX_SHADER))==0) return 0;
  if((fs = CreateShader(std::string(GetProjectTopDir() + "/src/graphics/src/graph_2d.f.glsl").c_str(), GL_FRAGMENT_SHADER))==0) return 0;
  program = glCreateProgram();
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
  if (!link_ok) 
  {
    fprintf(stderr, "glLinkProgram:");
    PrintLog(program);
    return 0;
  }
  glUseProgram(program);

  const char* attribute_name;
  attribute_name = "coord2d";
  attribute_coord2d = glGetAttribLocation(program, attribute_name);
  if (attribute_coord2d == -1)
  {
    fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
    return 0;
  }
  const char* uniform_name;
  uniform_name = "transform";
  uniform_transform = glGetUniformLocation(program, uniform_name);
  if (uniform_transform == -1)
  {
    fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
    return 0;
  }
  uniform_name = "color";
  uniform_color = glGetUniformLocation(program, uniform_name);
  if (uniform_color == -1)
  {
    fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
    return 0;
  }

  // init 2D shaders for nonuniform color
  GLuint nvs, nfs;
  /*!
   * /todo fix CreateShader graph_2d_nonuniform_color.v.glsl file path
   */
  if((nvs = CreateShader(std::string(GetProjectTopDir() + "/src/graphics/src/graph_2d_nonuniform_color.v.glsl").c_str(), GL_VERTEX_SHADER))==0) return 0;
  if((nfs = CreateShader(std::string(GetProjectTopDir() + "/src/graphics/src/graph_2d_nonuniform_color.f.glsl").c_str(), GL_FRAGMENT_SHADER))==0) return 0;
  program_varying_color = glCreateProgram();
  glAttachShader(program_varying_color, nvs);
  glAttachShader(program_varying_color, nfs);
  glLinkProgram(program_varying_color);
  glGetProgramiv(program_varying_color, GL_LINK_STATUS, &link_ok);
  if (!link_ok) 
  {
    fprintf(stderr, "glLinkProgram:");
    PrintLog(program_varying_color);
    return 0;
  }

  attribute_name = "coord2d_vc";
  attribute_coord2d_vc = glGetAttribLocation(program_varying_color, attribute_name);
  if (attribute_coord2d_vc == -1)
  {
    fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
    return 0;
  }
  uniform_name = "transform_vc";
  uniform_transform_vc = glGetUniformLocation(program_varying_color, uniform_name);
  if (uniform_transform_vc == -1)
  {
    fprintf(stderr, "Could not bind uniform %s\n", uniform_name);
    return 0;
  }
  attribute_name = "vcolor";
  attribute_color = glGetAttribLocation(program_varying_color, attribute_name);
  if (attribute_color == -1)
  {
    fprintf(stderr, "Could not bind attribute %s\n", attribute_name);
    return 0;
  }
 

  // generate buffers
  glGenBuffers(1, &border_vbo);
  glGenBuffers(1, &ticks_vbo);
  glGenBuffers(1, &grids_vbo);
  glGenBuffers(1, &markers_vbo);

  return 1;
}

void Plot2D::FreePlot()
{       
  glDeleteBuffers(1, &border_vbo);
  glDeleteBuffers(1, &ticks_vbo);
  glDeleteBuffers(1, &grids_vbo);
  glDeleteBuffers(1, &markers_vbo);
  std::cout << "Plot is freed." << std::endl;
}
void Plot2D::PlotOneCurve(uint r_indx, uint r_nc, GLfloat* r_color)
{
  uint num_plots = plots.size();
  glViewport(
    border_x,
    window_height/num_plots*r_indx + border_y,
    window_width - border_x*2,
    window_height/num_plots - border_y*2
  );  
  glScissor(
    border_x, 
    window_height/num_plots*r_indx + border_y,
    window_width - border_x*2,
    window_height/num_plots - border_y*2
  );

  glEnable(GL_SCISSOR_TEST);
  glm::mat4 transform = glm::translate(glm::scale(glm::mat4(1.0f), 
                        glm::vec3(plots[r_indx]->scale_x, plots[r_indx]->scale_y, 1)), 
                        glm::vec3(plots[r_indx]->offset_x, plots[r_indx]->offset_y, 0));
  glUniformMatrix4fv(uniform_transform, 1, GL_FALSE, glm::value_ptr(transform));
  glUniform4fv(uniform_color, 1, r_color);
  glBindBuffer(GL_ARRAY_BUFFER, (plots[r_indx]->curves)[r_nc]->vbo);
 
  glEnableVertexAttribArray(attribute_coord2d);
  glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glPointSize(3);

  if(!dynamic_cast<PhaseSpace*>((plots[r_indx]->curves)[r_nc]))
    glDrawArrays(GL_LINE_STRIP, 0, (plots[r_indx]->curves)[r_nc]->data_num);

  glDrawArrays(GL_POINTS, 0, (plots[r_indx]->curves)[r_nc]->data_num);
  glViewport(0, 0, window_width, window_height);
  glDisable(GL_SCISSOR_TEST);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Plot2D::PlotOneCurve(uint r_indx, uint r_nc)
{
  uint num_plots = plots.size();
  glViewport(
    border_x,
    window_height/num_plots*r_indx + border_y,
    window_width - border_x*2,
    window_height/num_plots - border_y*2
  );  
  glScissor(
    border_x, 
    window_height/num_plots*r_indx + border_y,
    window_width - border_x*2,
    window_height/num_plots - border_y*2
  );

  glEnable(GL_SCISSOR_TEST);
  glm::mat4 transform = glm::translate(glm::scale(glm::mat4(1.0f), 
                        glm::vec3(plots[r_indx]->scale_x, plots[r_indx]->scale_y, 1)), 
                        glm::vec3(plots[r_indx]->offset_x, plots[r_indx]->offset_y, 0));
  glUniformMatrix4fv(uniform_transform_vc, 1, GL_FALSE, glm::value_ptr(transform));

  Histogram2D* h2d = dynamic_cast<Histogram2D*>((plots[r_indx]->curves)[r_nc]);
  glBindBuffer(GL_ARRAY_BUFFER, h2d->color_vbo);
  glEnableVertexAttribArray(attribute_color);
  glVertexAttribPointer(attribute_color, 4, GL_FLOAT, GL_FALSE, 4*sizeof(GLfloat), 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_ARRAY_BUFFER, (plots[r_indx]->curves)[r_nc]->vbo);
  glEnableVertexAttribArray(attribute_coord2d_vc);
  glVertexAttribPointer(attribute_coord2d_vc, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glPointSize(5);
//  glDrawArrays(GL_LINE_STRIP, 0, 200);

  glDrawArrays(GL_POINTS, 0, (plots[r_indx]->curves)[r_nc]->data_num);
  glViewport(0, 0, window_width, window_height);
  glDisable(GL_SCISSOR_TEST);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Plot2D::PlotOneSubplot(uint r_indx)
{
  bool color_varying = false; 
  if(dynamic_cast<Histogram2D*>((plots[r_indx]->curves)[0]))
  {
    color_varying = true; 
    glUseProgram(program_varying_color);
  }
  if (color_varying)
  {
    for(uint i = 0; i < (plots[r_indx]->curves).size(); ++i)
      PlotOneCurve(r_indx, i);
    glUseProgram(program);
  }
  else
    for(uint i = 0; i < (plots[r_indx]->curves).size(); ++i)
      PlotOneCurve(r_indx, i, (plots[r_indx]->curves)[i]->color);
}

void Plot2D::DrawBorder(uint r_indx)
{
  float2 border[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  glBindBuffer(GL_ARRAY_BUFFER, border_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof border, border, GL_STATIC_DRAW);
  glm::mat4 transform = ViewportTransform(
    r_indx,
    border_x,
    border_y,
    window_width - border_x * 2,
    window_height/plots.size()- border_y * 2
  );
  glUniformMatrix4fv(uniform_transform, 1, GL_FALSE, glm::value_ptr(transform));

  glUniform4fv(uniform_color, 1, white);
  glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glLineWidth(1);
  glDrawArrays(GL_LINE_LOOP, 0, 4);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Plot2D::DrawTicksGrids(uint r_indx)
{
  /*------------------------------ y ticks & grids ------------------------------*/
  // desired space between ticks, in graph coordinates
  float tickspacingy = 0.1 * powf(10, -floor(log10(plots[r_indx]->scale_y))); 
  // left edge, in graph coordinates
  float low= -1.0 / plots[r_indx]->scale_y - plots[r_indx]->offset_y;                     
  // right edge, in graph coordinates
  float up = 1.0 / plots[r_indx]->scale_y - plots[r_indx]->offset_y;                     
//  std::cout << "tick spaceing=" << tickspacingy << ", low=" << low << ", up=" << up << std::endl;
  // index of left tick, counted from the origin
  int low_i = std::ceil(low/ tickspacingy);                      
  // index of right tick, counted from the origin
  int up_i = std::floor(up/ tickspacingy);                   
  // space between left edge of graph and the first tick
  float remy = low_i * tickspacingy - low;                    
  // first tick in device coordinates
  float firstticky = -1.0 + remy * plots[r_indx]->scale_y;                     
  // number of ticks to show
  int nticks = up_i - low_i + 1;                          

  float2 ticksy[2*nticks];
  float2 gridy[2*nticks];
  glUniform4fv(uniform_color, 1, white);
  for(int i = 0; i < nticks; i++) 
  {
    float y = firstticky + i * tickspacingy * plots[r_indx]->scale_y;
    float tickscale = ((i + low_i) % 5) ? 0.5 : 1.5;
    ticksy[i * 2].x = -1;
    ticksy[i * 2].y = y;
    ticksy[i * 2 + 1].x = -1 + tick_sz * tickscale * pixel_x;
    ticksy[i * 2 + 1].y = y;
    if(plots[r_indx]->show_grids)
    {
      gridy[i * 2].x = -1;
      gridy[i * 2].y = y;
      gridy[i * 2 + 1].x = 1;
      gridy[i * 2 + 1].y = y;
    }
    std::ostringstream ostr;
    ostr << std::setprecision(5)<< (i+low_i) * tickspacingy * plots[r_indx]->label_scale_y;
    std::string tmp_i = ostr.str();
    const char* char_i = tmp_i.c_str();
    glRasterPos2f(-1-tick_sz*pixel_x*tmp_i.size(), y-pixel_y*3.0);
    PrintBitmapString(GLUT_BITMAP_HELVETICA_10, char_i);
  }
  if(plots[r_indx]->show_grids)
  {
    glUniform4fv(uniform_color, 1, grey);
    glBindBuffer(GL_ARRAY_BUFFER, grids_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof gridy, gridy, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glLineWidth(1);
    glDrawArrays(GL_LINES, 0, 2*nticks);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  glUniform4fv(uniform_color, 1, white);
  glBindBuffer(GL_ARRAY_BUFFER, ticks_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof ticksy, ticksy, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glLineWidth(2.5);
  glDrawArrays(GL_LINES, 0, 2*nticks);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  /*------------------------------ x ticks & grids ------------------------------*/
  // desired space between ticks, in graph coordinates
  float tickspacing = 0.1 * powf(10, -floor(log10(plots[r_indx]->scale_x))); 
  // left edge, in graph coordinates
  float left = -1.0/plots[r_indx]->scale_x - plots[r_indx]->offset_x;                     
  // right edge, in graph coordinates
  float right = 1.0/plots[r_indx]->scale_x - plots[r_indx]->offset_x;                     
  // index of left tick, counted from the origin
  int left_i = ceil(left / tickspacing);                      
  // index of right tick, counted from the origin
  int right_i = floor(right / tickspacing);                   
  // space between left edge of graph and the first tick
  float rem = left_i * tickspacing - left;                    
  // first tick in device coordinates
  float firsttick = -1.0 + rem * plots[r_indx]->scale_x;                     
  // number of ticks to show
  nticks = right_i - left_i + 1;                          

  glUniform4fv(uniform_color, 1, white);
  float2 ticks[2*nticks];
  float2 gridx[2*nticks];
  for(int i = 0; i < nticks; i++) 
  {
    float x = firsttick + i * tickspacing * plots[r_indx]->scale_x;
    float tickscale = ((i + left_i) % 5) ? 0.5 : 1.5;
    ticks[i * 2].x = x;
    ticks[i * 2].y = -1;
    ticks[i * 2 + 1].x = x;
    ticks[i * 2 + 1].y = -1 + tick_sz * tickscale * pixel_y;
    if(plots[r_indx]->show_grids)
    {
      gridx[i * 2].x = x;
      gridx[i * 2].y = -1;
      gridx[i * 2 + 1].x = x;
      gridx[i * 2 + 1].y = 1;
    }
    std::ostringstream ostr;
    ostr << (i+left_i) * tickspacing * plots[r_indx]->label_scale_x;
    std::string tmp_i = ostr.str();
    const char* char_i = tmp_i.c_str();
    glRasterPos2f(x-pixel_x*2*tmp_i.size(), -1-tick_sz*pixel_y*1.5);
    PrintBitmapString(GLUT_BITMAP_HELVETICA_10, char_i);
  }
  if(plots[r_indx]->show_grids)
  {
    glUniform4fv(uniform_color, 1, grey);
    glBindBuffer(GL_ARRAY_BUFFER, grids_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof gridx, gridx, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glLineWidth(1);
    glDrawArrays(GL_LINES, 0, 2*nticks);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  glUniform4fv(uniform_color, 1, white);
  glBindBuffer(GL_ARRAY_BUFFER, ticks_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof ticks, ticks, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glLineWidth(2.5);
  glDrawArrays(GL_LINES, 0, nticks*2);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

glm::mat4 Plot2D::ViewportTransform(uint r_indx, float r_x, float r_y, 
                                    float r_width, float r_height)
{
  float w_height = window_height/plots.size();
  float offset_x = (2.0 * r_x + (r_width - window_width))/ window_width;
  float offset_y = (2.0 * r_y + (r_height - window_height) + 2*w_height*r_indx) / window_height ;
  float scale_x = r_width / window_width;
  float scale_y = r_height / window_height;

  return glm::scale(glm::translate(glm::mat4(1), glm::vec3(offset_x, offset_y, 0)), glm::vec3(scale_x, scale_y, 1)); 
}

void Plot2D::DrawLastDataTracker(uint r_indx)
{
//  if((plots[r_indx]->curves).empty()) return;
  for(uint i = 0; i < (plots[r_indx]->curves).size(); ++i)
    if((plots[r_indx]->curves)[i]->data_tracker_on)
      DrawOneLastDataTracker(r_indx, i, (plots[r_indx]->curves)[i]->color);
}

void Plot2D::DrawOneLastDataTracker(uint r_pindx, uint r_cindx, GLfloat* r_color)
{
  glUniform4fv(uniform_color, 1, r_color);
  std::ostringstream ostr_l;
  ostr_l << "(" << (plots[r_pindx]->curves)[r_cindx]->last_data.x * plots[r_pindx]->label_scale_x << ", " 
                << (plots[r_pindx]->curves)[r_cindx]->last_data.y * plots[r_pindx]->label_scale_y << ")";
  std::string tmp_str = ostr_l.str();
  const char* char_str = tmp_str.c_str();
  float last_data_x = 1-pixel_x*tmp_str.size()*6;
  float last_data_y = 1-pixel_y*(r_cindx+2)*12;
  glRasterPos2f(last_data_x, last_data_y);
  PrintBitmapString(GLUT_BITMAP_HELVETICA_10, char_str);
}

void Plot2D::DrawLabels(uint r_indx)
{
  for(uint i = 0; i < (plots[r_indx]->curves).size(); ++i)
    if((plots[r_indx]->curves)[i]->label_y != "")
      DrawOneLabel(r_indx, i, (plots[r_indx]->curves)[i]->color);
}

void Plot2D::DrawOneLabel(uint r_pindx, uint r_cindx, GLfloat* r_color)
{
  glUniform4fv(uniform_color, 1, r_color);
  const char* char_str = ((plots[r_pindx]->curves)[r_cindx]->label_y).c_str();
  float pos_x = 0;//-1+pixel_x*((plots[r_pindx]->curves)[r_cindx]->label_y).size()*6;
  float pos_y = 1-pixel_y*(r_cindx+2)*12;
  glRasterPos2f(pos_x, pos_y);
  PrintBitmapString(GLUT_BITMAP_HELVETICA_10, char_str);
}

void Plot2D::DrawMarkers(uint r_indx)
{
  if((plots[r_indx]->xmarker).empty() && (plots[r_indx]->ymarker).empty()) return;
  if(!(plots[r_indx]->xmarker).empty())
  {
    const size_t xmarker_num = (plots[r_indx]->xmarker).size();
    float2 marker_x[2*xmarker_num];
    double left, rem, coord;
    for(int i = 0; i < xmarker_num; ++i)
    {
//      std::cout << "put an X marker at " << (plots[r_indx]->xmarker)[i] << ", size of marker_x is " << sizeof(marker_x) << std::endl;
      left = -1.0/plots[r_indx]->scale_x - plots[r_indx]->offset_x; // in real value coordinate
      rem = (plots[r_indx]->xmarker)[i]-left;
      coord = -1.0 + rem * plots[r_indx]->scale_x; 
      marker_x[i * 2].x = coord;
      marker_x[i * 2].y = -1;
      marker_x[i * 2 + 1].x = coord;
      marker_x[i * 2 + 1].y = 1;
    }
    glUniform4fv(uniform_color, 1, white);
    glBindBuffer(GL_ARRAY_BUFFER, markers_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(marker_x), marker_x, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glLineWidth(1);
    glDrawArrays(GL_LINES, 0, 2*xmarker_num);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  if(!(plots[r_indx]->ymarker).empty())
  {
    const size_t ymarker_num = (plots[r_indx]->ymarker).size();
    float2 marker_y[2*ymarker_num];
    double low, rem, coord;
    for(int i = 0; i < ymarker_num; ++i)
    {
      std::cout << "put an Y marker at " << (plots[r_indx]->ymarker)[i] << std::endl;
      low = -1.0/plots[r_indx]->scale_y - plots[r_indx]->offset_y; // in real value coordinate
      rem = (plots[r_indx]->ymarker)[i]-low;
      coord = -1.0 + rem * plots[r_indx]->scale_y; 
      marker_y[i * 2].x = -1;
      marker_y[i * 2].y = coord;
      marker_y[i * 2 + 1].x = 1;
      marker_y[i * 2 + 1].y = coord;
    }
    glUniform4fv(uniform_color, 1, white);
    glBindBuffer(GL_ARRAY_BUFFER, markers_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof marker_y, marker_y, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glLineWidth(1);
    glDrawArrays(GL_LINES, 0, 2*ymarker_num);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

}

void Plot2D::Display()
{
  SetWindowSize(glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT);
  if(plots.empty()) return;
  pixel_x = 2.0 / (window_width - border_x * 2);
  pixel_y = 2.0 / (window_height/plots.size()- border_y * 2);
  for(uint i = 0; i < plots.size(); ++i)
  {
    if((plots[i]->curves).empty()) continue;
    PlotOneSubplot(i);
    DrawBorder(i);
    DrawTicksGrids(i);
    DrawLastDataTracker(i);
    DrawLabels(i);
    DrawMarkers(i);
  }
  glDisableVertexAttribArray(attribute_coord2d);
  glutSwapBuffers();
}

void Plot2D::Update()
{
  for(uint i = 0; i < plots.size(); ++i)
    plots[i]->UpdateSubplot();
}
