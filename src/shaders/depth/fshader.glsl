#version 330 core

out float color;

void main(){
  color = gl_FragCoord.z;
}
