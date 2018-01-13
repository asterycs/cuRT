#version 330 core

out vec2 UV;

void main() {
  float xCoord = (gl_VertexID == 2) ?  3.0 : -1.0;
  float yCoord = (gl_VertexID == 1) ? -3.0 :  1.0;
  UV.x = (xCoord + 1.f) * 0.5f;
  UV.y = (yCoord + 1.f) * 0.5f;

  gl_Position = vec4(xCoord, yCoord, 0.f, 1.f);
}
