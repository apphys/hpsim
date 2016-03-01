#ifndef GRAPHICS_COMMON_FUNC_H
#define GRAPHICS_COMMON_FUNC_H

#ifdef NOGLEW
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#else
#include "glew.h"
#endif
#include <GL/glut.h>

extern GLfloat yellow[];
extern GLfloat white[];
extern GLfloat grey[];
extern GLfloat black[];
extern GLfloat red[];
extern GLfloat green[];
extern GLfloat blue[];
extern GLfloat skyblue[];
extern GLfloat magenta[];
extern GLfloat cyan[];
extern GLfloat orange[];
char* FileRead(const char* filename);
void PrintLog(GLuint object);
GLint CreateShader(const char* filename, GLenum type);
void PrintBitmapString(void* font, const char* s);
uint CreateVBO(uint r_sz);
float Lerp(float r_a, float r_b, float r_t);
void ColorRamp(float r_t, float* r_r);
void ColorRampWhiteBackGround(float r_t, float* r_r, double);

#endif
