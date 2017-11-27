#version 330 core

in vec4 position;

out vec2 ftexCoord;

void main() {
  gl_Position = vec4(position.xy, 0.f, 1.0);
  ftexCoord = position.zw;
}
