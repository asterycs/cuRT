#ifndef ISPCRENDERER_HPP
#define ISPCRENDERER_HPP

#include "Light.hpp"
#include "Model.hpp"
#include "GLTexture.hpp"
#include "Camera.hpp"

class ISPCRenderer
{
public:
  ISPCRenderer();
  ~ISPCRenderer();
  
  void renderToCanvas(GLTexture& canvas, const Camera& camera, Model& model, Light& light);


};

#endif // ISPCRENDERER_HPP
