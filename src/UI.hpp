#ifndef UI_HPP
#define UI_HPP

#include "GLTexture.hpp"

class GLFWwindow;

class UI
{
public:
  UI();
  ~UI();

  void init(GLFWwindow* window);

  void draw();
  void resize(const glm::ivec2 newSize);
private:
  GLTexture fontTexture;
};

#endif /* UI_HPP_ */
