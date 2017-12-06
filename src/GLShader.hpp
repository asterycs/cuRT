#ifndef GLSHADER_HPP
#define GLSHADER_HPP

#include <string>
#include <vector>

#include <GL/glew.h>
#include <GL/gl.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Utils.hpp"

class GLShader
{
public:
  GLShader();
  GLShader(GLShader const& that) = delete;
  void operator=(GLShader& that) = delete;
  ~GLShader();

  void loadShader(const std::string& vertex_path, const std::string& fragment_path);
  bool isLoaded() const;
  void bind();
  void unbind();
  
  void updateUniform3fv(const std::string& identifier, const glm::fvec3& value);
  void updateUniformMat4f(const std::string& identifier, const glm::fmat4& mat);
  void updateUniformMat3f(const std::string& identifier, const glm::fmat3& mat);
  void updateUniformMat2f(const std::string& identifier, const glm::fmat2& mat);
  void updateUniform1i(const std::string& identifier, const int value);
  
  GLint getAttribLocation(const std::string& identifier);

private:
  GLuint program;
  bool isOperational;
};

#endif // GLSHADER_HPP
