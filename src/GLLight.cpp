#include "GLLight.hpp"

#include <vector>

GLLight::GLLight() :
  light(),
  shadowMap(glm::ivec2(SHADOWMAP_WIDTH, SHADOWMAP_WIDTH)),
  depthMVP()
{

}

GLLight::~GLLight()
{

}

void GLLight::load(const Light& light)
{
  clear();


  this->light = light;

  Material material;
  {
    material.colorAmbient   = glm::vec3(1.0f, 1.0f, 1.0f);
    material.colorDiffuse   = glm::vec3(1.0f, 1.0f, 1.0f);
    material.colorEmission  = glm::vec3(1.0f, 1.0f, 1.0f);
    material.colorSpecular  = glm::vec3(1.0f, 1.0f, 1.0f);
    material.colorShininess = glm::vec3(1.0f, 1.0f, 1.0f);
  }

  std::vector<MeshDescriptor> meshDescriptors;

  meshDescriptors.push_back(MeshDescriptor(0, 2, material));

  const glm::vec2& lightSize = light.getSize();

  const std::vector<glm::vec4> contour = {
    glm::vec4(-lightSize.x, lightSize.y, 0.0, 1.0),
    glm::vec4(lightSize.x, -lightSize.y, 0.0, 1.0),
    glm::vec4(-lightSize.x, -lightSize.y, 0.0, 1.0),

    glm::vec4(lightSize.x, lightSize.y, 0.0, 1.0),
    glm::vec4(lightSize.x, -lightSize.y, 0.0, 1.0),
    glm::vec4(-lightSize.x, lightSize.y, 0.0, 1.0),
  };

  std::vector<Vertex> vertices;

  for (auto& c : contour)
  {
    const glm::mat4 M = light.getModelMat(); // Camera to world
    const glm::mat3 N = glm::transpose(glm::inverse(glm::mat3(light.getModelMat()))); // Normal to world

    Vertex v;
    v.p = glm::vec3(M * c);
    v.n = glm::vec3(N * glm::vec3(0.0f, 0.0f, 1.f));
    vertices.push_back(v);
  }

  unsigned int nTriangles = GLuint(vertices.size() / 3);

  std::vector<Triangle> triangles;

  for (unsigned int i = 0; i < nTriangles; ++i)
    triangles.push_back(Triangle(vertices[i*3], vertices[i*3+1], vertices[i*3+2]));
    
  glm::mat4 depthProjectionMatrix = glm::perspective(glm::half_pi<float>(), (float) light.getSize().x / (float) light.getSize().y, 0.001f, 10.f);
  depthMVP = depthProjectionMatrix * glm::inverse(light.getModelMat());

  finalizeLoad(triangles, meshDescriptors);
}

const Light& GLLight::getLight() const
{
  return light;
}

glm::mat4 GLLight::getDepthBiasMVP() const
{
  glm::mat4 bias(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.5, 0.5, 0.5, 1.0
  );
  
  glm::mat4 depthBiasMVP = bias * depthMVP;
  
  return depthBiasMVP;
}

glm::mat4 GLLight::getDepthMVP() const
{
  return depthMVP;
}

const GLShadowMap& GLLight::getShadowMap() const
{
  return shadowMap;
}


