#version 330 core

uniform mat4 posToCamera;
uniform mat4 normalToCamera;
uniform mat4 biasedDepthToLight;

layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec3 normal;

out vec3 vnormal;
out vec4 shadowCoord;
out vec4 worldPos;

void main() {
  gl_Position =  posToCamera * vec4(vertexPosition, 1.0);
  worldPos = vec4(vertexPosition, 1.0);
  vnormal = normal;
  shadowCoord = biasedDepthToLight * vec4(vertexPosition, 1.0);
}
