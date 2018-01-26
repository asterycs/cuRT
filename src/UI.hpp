#ifndef UI_HPP
#define UI_HPP

#include "GLTexture.hpp"

class GLFWwindow;

enum ActiveRenderer {
  GL
#ifdef ENABLE_CUDA
  , RAYTRACER,
  PATHTRACER
#endif
};

enum DebugMode
{
  DEBUG_RAYTRACE,
  DEBUG_PATHTRACE,
  NONE
};

class UI
{
public:
  UI();
  ~UI();

  void init(GLFWwindow* window);

  void draw(const enum ActiveRenderer activeRenderer, const enum DebugMode debugMode);
  void resize(const glm::ivec2 newSize);
  float getDTime();
private:
  GLTexture fontTexture;
};

#endif /* UI_HPP_ */
