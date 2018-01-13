#include "UI.hpp"

#include "imgui.h"

#include "Utils.hpp"

UI::UI(const GLContext& glcontext)
{
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize.x = WWIDTH;
  io.DisplaySize.y = WHEIGHT;
  //io.RenderDrawListsFn = MyRenderFunction;

  unsigned char* pixels;
  int width, height;
  io.Fonts->GetTexDataAsRGBA32(pixels, &width, &height);
  GLTexture texture = glcontext.createTexture(pixels, width, height, TEXTURE_TYPE_RGBA);
  io.Fonts->TexID = (void*)texture;
}

UI::~UI()
{
  // TODO Auto-generated destructor stub
}

