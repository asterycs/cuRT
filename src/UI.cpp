#include "UI.hpp"

#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include "imgui.h"

#include "Utils.hpp"
#include "imgui_impl_glfw_gl3.h"

UI::UI()
{

}

void UI::init(GLFWwindow* window)
{
  ImGui_ImplGlfwGL3_Init(window, false);
  ImGui::StyleColorsClassic();
}

UI::~UI()
{

}

void UI::draw()
{
  ImGui_ImplGlfwGL3_NewFrame();
  ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
  bool showDemoWindow = true;
  ImGui::ShowDemoWindow(&showDemoWindow);
  ImGui::Render();
}

void UI::resize(const glm::ivec2 newSize)
{
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize.x = newSize.x;
  io.DisplaySize.y = newSize.y;
}

