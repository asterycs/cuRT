#version 330 core

in vec2 UV;

out vec4 color;

uniform sampler2D texture;

void main(){
  color = texture2D(texture, UV);
}
