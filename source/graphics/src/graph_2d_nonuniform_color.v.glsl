#version 120

attribute vec2 coord2d_vc;
uniform mat4 transform_vc;
attribute vec4 vcolor;
varying vec4 fcolor;

void main(void) {
  gl_Position = transform_vc * vec4(coord2d_vc.xy, 0, 1);
  fcolor = vcolor;
}
