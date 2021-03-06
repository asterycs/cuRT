#ifndef GLCONTEXT_HPP
#define GLCONTEXT_HPP

#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include <ft2build.h>
#include FT_FREETYPE_H

#include "Camera.hpp"
#include "GLDrawable.hpp"
#include "GLLight.hpp"
#include "GLModel.hpp"
#include "GLTexture.hpp"
#include "GLShader.hpp"
#include "GLShadowMap.hpp"
#include "UI.hpp"



class GLContext
{
public:
  GLContext();
  ~GLContext();

  void draw(const GLModel& scene, const GLLight& light, const Camera& mvp);
  void draw(const GLModel& scene, const GLLight& light, const Camera& mvp, const Node& debugNode);
  void draw(const GLTexture& canvas);
  void draw(const std::vector<glm::fvec3>& points, const Camera& camera);
  void draw(const AABB& box, const Camera& camera);

  void drawUI(const enum ActiveRenderer activeRenderer, const enum DebugMode debugMode);
  bool UiWantsMouseInput();
  void resize(const glm::ivec2& newSize);
  bool shadersLoaded() const;
  
  void showWindow();

  glm::ivec2 getSize() const;

  void clear();
  void swapBuffers();
  bool isAlive();
  
  float getDTime();
  glm::ivec2 getCursorPos();
  bool isKeyPressed(const int glfwKey, const int modifiers) const;
  
  float getTime() const;

private:
  void drawModel(const GLModel& model, const Camera& camera, const GLLight& light);
  void drawNodeTriangles(const GLModel& model, const GLLight& light, const Camera& camera, const Node& node);
  void drawLight(const GLLight& light, const Camera& camera);
  void drawShadowMap(const GLLight& light);
  void updateShadowMap(const GLDrawable& model, const GLLight& light);
  void updateUniformMat4f(const glm::mat4& mat, const std::string& identifier);
  void updateUniform3fv(const glm::vec3& vec, const std::string& identifier);

  GLuint loadShader(const char *vertex_path, const char *fragment_path);
  GLShader modelShader;
  GLShader lightShader;
  GLShader canvasShader;
  GLShader textShader;
  GLShader depthShader;
  GLShader lineShader;
  GLShader triShader;
  GLFWwindow* window;

  glm::ivec2 size;

  UI ui;
};

#endif // GLCONTEXT_HPP
