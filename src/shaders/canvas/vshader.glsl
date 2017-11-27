#version 330 core

layout(location = 0) in vec3 position;
layout(location = 2) in vec2 tex;

out vec2 UV;

void main() {
  gl_Position = vec4(position.xyz, 1.0);
  UV = tex;
}
