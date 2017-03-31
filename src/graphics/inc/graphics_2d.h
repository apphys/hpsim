#ifndef GRAPHICS_2D_H
#define GRAPHICS_2D_H

// WARNING: glew.h has to be included before <cuda_gl_interop.h>
#include <GL/glew.h> 
#include <GL/glut.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <GL/glm/glm.hpp>

struct Curve2D
{
  Curve2D();
  Curve2D(double*, double*, uint, GLfloat*, bool r_auto_update = true, 
    bool r_data_tracker_on = true);
  void Initialize(double*, double*, uint, GLfloat*, bool r_auto_update = true, 
    bool r_data_tracker_on = true);
  virtual ~Curve2D();
  virtual void UpdateData(); // call kernel to set data on device
  double GetMaxX() const
  {
    return maxmin->x;
  }
  double GetMinX() const
  {
    return maxmin->y;
  }
  double GetMaxY() const
  {
    return maxmin->z;
  }
  double GetMinY() const
  {
    return maxmin->w;
  }
  uint GetGLBuffer() const
  {
    return vbo;
  }
  void SetMaxMin(double r_x_min, double r_x_max, double r_y_min, double r_y_max)
  {
    auto_update = false;
    maxmin->y = r_x_min;
    maxmin->x = r_x_max;
    maxmin->w = r_y_min;
    maxmin->z = r_y_max;
  }
  void TurnOffAutoUpdate()
  {
    auto_update = false;
  }
  void TurnOnAutoUpdate()
  {
    auto_update = true;
  }
  void SetLabelX(std::string r_label)
  {
    label_x = r_label;  
  }  
  void SetLabelY(std::string r_label)
  {
    label_y = r_label;  
  }

  double* input_x;
  double* input_y;
  uint data_num;
  GLfloat* color;
  float2* data;         // device pointer to data
  uint blck_num;        // num of blcks in a grid
  uint thrd_num;        // num of thrds in a block
  uint vbo;             // vertex buffer object
  struct cudaGraphicsResource* cuda_vbo;
  double4* maxmin;        // host ptr to the mapped mem
  double2 last_data;     // host data
  double* partial_x;  // size of num of blocks+1
  double* partial_y;  // size of num of blocks+1
  double* partial_x2;  // size of num of blocks+1
  double* partial_y2;  // size of num of blocks+1
  bool auto_update;
  bool data_tracker_on;
  std::string label_x, label_y;
};

struct Histogram2D: public Curve2D
{
  Histogram2D(double*, double*, uint*, uint, uint r_bin_num_x, uint r_bin_num_y);
  Histogram2D(double*, double*, uint*, uint, double*, double*, double*, 
    double*, uint, uint);
  void InitHistogram2D();
  ~Histogram2D();
  void UpdateData();
  
  uint raw_data_num;
  uint bin_num_x;
  uint bin_num_y;
  double* avg_x;
  double* avg_y;
  double* sig_x;
  double* sig_y;
  uint* input_loss;
  uint* hist;
  uint color_vbo;
//  struct cudaGraphicsResource* cuda_color_vbo;
};

struct PhaseSpace : public Curve2D
{
  PhaseSpace(double*, double*, uint*, uint, GLfloat*, bool r_auto_update = true);
  ~PhaseSpace();
  void UpdateData();
  uint* input_loss;
  double* tmp_x;
  double* tmp_y;
  uint* tmp_loss;
};

struct Histogram : public Curve2D
{
  Histogram(double*, uint*, uint, double, double, GLfloat*, 
    uint r_bin_num = 64 * 2);
  ~Histogram();
  void UpdateData();
  double max;
  double min;
  uint hist_data_num; // num of original data 
  uint* input_loss;
  uint* hist;
  uint* partial_hist;
};

struct Subplot2D
{
  Subplot2D();
  void AddCurve(Curve2D*);
  void UpdateSubplot();
  void AddXMarker(double r_x_marker)
  {
    xmarker.push_back(r_x_marker);
  }
  void AddYMarker(double r_y_marker)
  {
    ymarker.push_back(r_y_marker);
  }
  void ResetScale()
  {
    scale_ratio = 1;
  }
  void ResetOffSet()
  {
    offset_mouse_x = 0.0;
    offset_mouse_y = 0.0;
  }     
  void TurnOffAutoUpdate()
  {
    std::vector<Curve2D*>::iterator iter = curves.begin();
    for(; iter != curves.end(); ++iter)
      (*iter)->TurnOffAutoUpdate();
  }  
  void TurnOnAutoUpdate()
  {
    std::vector<Curve2D*>::iterator iter = curves.begin();
    for(; iter != curves.end(); ++iter)
      (*iter)->TurnOnAutoUpdate();
  }  
  void SetMaxMin(double r_x_min, double r_x_max, double r_y_min, double r_y_max)
  {
    std::vector<Curve2D*>::iterator iter = curves.begin();
    for(; iter != curves.end(); ++iter)
      (*iter)->SetMaxMin(r_x_min / label_scale_x, r_x_max / label_scale_x, 
                        r_y_min / label_scale_y, r_y_max / label_scale_y);
  }
  void SetLabelScaleX(float r_scale)
  {
    label_scale_x = r_scale;
  }
  void SetTickLabelScaleY(float r_scale)
  {
    label_scale_y = r_scale;
  }
  std::vector<Curve2D*> curves;
  std::vector<double> xmarker;
  std::vector<double> ymarker;
  float offset_x, offset_y, scale_x, scale_y;
  float offset_x_init, offset_y_init, scale_x_init, scale_y_init;
  float scale_ratio, offset_mouse_x, offset_mouse_y;
  float label_scale_x, label_scale_y;
  bool show_grids;
};

struct Plot2D
{
  int InitPlot();
  void FreePlot();
  void PlotOneSubplot(uint);
  void PlotOneCurve(uint, uint, GLfloat*);
  void PlotOneCurve(uint, uint);
  void DrawBorder(uint);
  void DrawTicksGrids(uint);
  void DrawLastDataTracker(uint);
  void DrawOneLastDataTracker(uint, uint, GLfloat*);
  void DrawMarkers(uint);
  void DrawLabels(uint);
  void DrawOneLabel(uint, uint, GLfloat*);
  glm::mat4 ViewportTransform(uint, float, float, float, float);
  void Display();
  void Update();
  void AddSubplot(Subplot2D* r_splot)
  {
    plots.push_back(r_splot);
  }
  void SetWindowSize(int r_w, int r_h)
  {
    window_width = r_w; window_height = r_h; 
  }
  void SetBorderWidth(int r_x, int r_y)
  {
    border_x = r_x; border_y = r_y;
  }
  void SetTickSize(int r_tick)
  {
    tick_sz = r_tick;
  }

  std::vector<Subplot2D*> plots;
  GLuint program;
  GLuint program_varying_color;
  GLint attribute_coord2d;
  GLint attribute_coord2d_vc;
  GLint uniform_color;
  GLint attribute_color;
  GLint uniform_transform;
  GLint uniform_transform_vc;
  uint border_vbo;
  uint ticks_vbo;
  uint grids_vbo;
  uint markers_vbo;
  int window_width, window_height;
  int border_x, border_y;
  int tick_sz;
  float pixel_x, pixel_y;
};

#endif
