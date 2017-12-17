#version 330 core

struct Material
{
  vec3 colorDiffuse;
  vec3 colorAmbient;
};

uniform Material material;
uniform sampler2D shadowMap;

uniform vec3 lightPos;
uniform vec3 lightNormal;

in vec3 vnormal;
in vec4 shadowCoord;
in vec4 worldPos;

out vec3 color;

void main(){
  vec3 toLight = normalize(lightPos - worldPos.xyz);
  float cosL = clamp(dot(-lightNormal, toLight), 0.f, 1.f);
  float cosT = clamp(dot(vnormal, toLight), 0.f, 1.f);
  bool lightVisible = false;

  if (cosL > 0)
    lightVisible = true;

  float epsilon = 0.001 * tan(acos(cosT));
  epsilon = clamp(epsilon, 0.f, 0.00001f);

  float visibility = 1.0;
  vec4 shadowCoordNorm = shadowCoord / shadowCoord.w;

  if (shadowCoordNorm.x >= 0 && shadowCoordNorm.x <= 1 && shadowCoordNorm.y >= 0 && shadowCoordNorm.y <= 1 && lightVisible)
  {
      if (texture(shadowMap, shadowCoordNorm.xy).r  <  shadowCoordNorm.z - epsilon){
        visibility = 0.5;
      }
  }

  vec3 ambient = material.colorAmbient * 0.25f;
  vec3 diffuse = visibility * material.colorDiffuse * cosT * cosL;
  color = ambient + diffuse;
}
