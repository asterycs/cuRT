#version 330 core

in vec2 ftexCoord;
uniform sampler2D texture;

out vec4 color;

void main(void) {
  color = vec4(1.f, 1.f, 1.f, texture2D(texture, ftexCoord).r);
}
