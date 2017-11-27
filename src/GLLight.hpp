#ifndef GLLIGHT_HPP
#define GLLIGHT_HPP

#include "GLDrawable.hpp"
#include "Light.hpp"
#include "GLShadowMap.hpp"

#define SHADOWMAP_WIDTH 1024

class GLLight : public GLDrawable
{
public:
  GLLight();
  GLLight(const GLDrawable& that) = delete;
  GLLight& operator=(const GLDrawable& that) = delete;
  ~GLLight();

  void load(const Light& light);

  const Light& getLight() const;
  
  glm::mat4 getDepthBiasMVP() const;
  glm::mat4 getDepthMVP() const;
  const GLShadowMap& getShadowMap() const;  


private:
  Light light;
  GLShadowMap shadowMap;
  glm::mat4 depthMVP;
};

#endif
