#ifndef ISPCRENDERER_HPP
#define ISPCRENDERER_HPP

#include "Light.hpp"
#include "Model.hpp"
#include "GLCanvas.hpp"
#include "Camera.hpp"

class ISPCRenderer
{
public:
  ISPCRenderer();
  ~ISPCRenderer();
  
  void renderToCanvas(GLCanvas& canvas, const Camera& camera, Model& model, Light& light);


};

#endif // ISPCRENDERER_HPP
