#include "UI.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include "imgui.h"

#include "Utils.hpp"

UI::UI()
{
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize.x = WWIDTH;
  io.DisplaySize.y = WHEIGHT;
  //io.RenderDrawListsFn = MyRenderFunction;

  unsigned char* pixels;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
  fontTexture = texture(pixels, glm::ivec2(width, height));
}

UI::~UI()
{
  // TODO Auto-generated destructor stub
}

