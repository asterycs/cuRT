#version 330 core

uniform mat4 posToCamera;

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 normal;

void main() {
  gl_Position =  posToCamera * vec4(vertexPosition, 1.0);
}
