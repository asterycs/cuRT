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

void UI::draw(const enum ActiveRenderer activeRenderer, const enum DebugMode debugMode)
{
  ImGui_ImplGlfwGL3_NewFrame();

  const float DISTANCE = 10.0f;
  static int corner = 0;
  ImVec2 window_pos = ImVec2((corner & 1) ? ImGui::GetIO().DisplaySize.x - DISTANCE : DISTANCE, (corner & 2) ? ImGui::GetIO().DisplaySize.y - DISTANCE : DISTANCE);
  ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
  ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.3f));
  if (ImGui::Begin("Info", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings))
  {
      ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

      switch (activeRenderer)
      {
        case GL:
          ImGui::Text("Renderer (enter): OpenGL");
          break;

#ifdef ENABLE_CUDA
        case RAYTRACER:
          ImGui::Text("Renderer (enter): Raytracer");
          break;

        case PATHTRACER:
          ImGui::Text("Renderer (enter): Pathtracer");
          break;
#endif

        default:
          break;
      }

      switch (debugMode)
      {
        case DEBUG_RAYTRACE:
          ImGui::Text("Debug (ctrl + D): Raytrace");
          break;
        case DEBUG_PATHTRACE:
          ImGui::Text("Debug (ctrl + D): Pathtrace");
          break;

        default:
          ImGui::Text("Debug (ctrl + D): None");
          break;
      }

      ImGui::Text("Open model: O");
      ImGui::Text("Open scene file: Ctrl+O");
      ImGui::Text("Save scene file: Ctrl+S");


      if (ImGui::BeginPopupContextWindow())
      {
          if (ImGui::MenuItem("Top-left", NULL, corner == 0)) corner = 0;
          if (ImGui::MenuItem("Top-right", NULL, corner == 1)) corner = 1;
          if (ImGui::MenuItem("Bottom-left", NULL, corner == 2)) corner = 2;
          if (ImGui::MenuItem("Bottom-right", NULL, corner == 3)) corner = 3;
          ImGui::EndPopup();
      }
      ImGui::End();
  }
  ImGui::PopStyleColor();

  ImGui::Render();
}

float UI::getDTime()
{
  return ImGui::GetIO().DeltaTime;
}

void UI::resize(const glm::ivec2 newSize)
{
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize.x = newSize.x;
  io.DisplaySize.y = newSize.y;
}

