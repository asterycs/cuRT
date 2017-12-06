#include "GLShader.hpp"

GLShader::GLShader() :
  program(0),
  isOperational(false)
{

}


GLShader::~GLShader()
{
}

void GLShader::loadShader(const std::string& vertex_path, const std::string& fragment_path) {
    GLuint vertShader, fragShader;
    GL_CHECK(vertShader = glCreateShader(GL_VERTEX_SHADER));
    GL_CHECK(fragShader = glCreateShader(GL_FRAGMENT_SHADER));

    // Read shaders
    std::string vertShaderStr = readFile(vertex_path);
    std::string fragShaderStr = readFile(fragment_path);
    const char *vertShaderSrc = vertShaderStr.c_str();
    const char *fragShaderSrc = fragShaderStr.c_str();

    GLint result = GL_FALSE;
    int logLength;

    // Compile vertex shader
    GL_CHECK(glShaderSource(vertShader, 1, &vertShaderSrc, NULL));
    GL_CHECK(glCompileShader(vertShader));

    // Check vertex shader
    GL_CHECK(glGetShaderiv(vertShader, GL_COMPILE_STATUS, &result));
    GL_CHECK(glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &logLength));
    std::vector<char> vertShaderError((logLength > 1) ? logLength : 1);
    GL_CHECK(glGetShaderInfoLog(vertShader, logLength, NULL, &vertShaderError[0]));
    
    if (logLength > 0)
    {
      std::cout << &vertShaderError[0];
      std::cout << "Vertex shader compilation failed" << std::endl << std::endl;
      return;
    }

    // Compile fragment shader
    GL_CHECK(glShaderSource(fragShader, 1, &fragShaderSrc, NULL));
    GL_CHECK(glCompileShader(fragShader));

    // Check fragment shader
    GL_CHECK(glGetShaderiv(fragShader, GL_COMPILE_STATUS, &result));
    GL_CHECK(glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &logLength));
    std::vector<char> fragShaderError((logLength > 1) ? logLength : 1);
    GL_CHECK(glGetShaderInfoLog(fragShader, logLength, NULL, &fragShaderError[0]));
    
    if (logLength > 0)
    {
      std::cout << &fragShaderError[0];
      std::cout << "Fragment shader compilation failed" << std::endl << std::endl;
      return;
    }

    GLuint glProgram;
    GL_CHECK(glProgram = glCreateProgram());
    GL_CHECK(glAttachShader(glProgram, vertShader));
    GL_CHECK(glAttachShader(glProgram, fragShader));
    GL_CHECK(glLinkProgram(glProgram));

    GL_CHECK(glGetProgramiv(glProgram, GL_LINK_STATUS, &result));
    GL_CHECK(glGetProgramiv(glProgram, GL_INFO_LOG_LENGTH, &logLength));
    std::vector<char> programError( (logLength > 1) ? logLength : 1 );
    GL_CHECK(glGetProgramInfoLog(glProgram, logLength, NULL, &programError[0]));
    
    if (logLength > 0)
    {
      std::cout << &programError[0];
      std::cout << "Shader linking failed" << std::endl << std::endl;
      return;
    }
    
    this->program = glProgram;

    GL_CHECK(glDeleteShader(vertShader));
    GL_CHECK(glDeleteShader(fragShader));
    
    isOperational = true;

    return;
}

bool GLShader::isLoaded() const
{
  return isOperational;
}

void GLShader::bind()
{
  GL_CHECK(glUseProgram(program));
}

void GLShader::unbind()
{
  GL_CHECK(glUseProgram(0));
}

void GLShader::updateUniform3fv(const std::string& identifier, const glm::fvec3& value)
{
  GLint id;
  GL_CHECK(id = glGetUniformLocation(program, identifier.c_str()));
  if (id == -1)
    std::cerr << "Identifier \"" << identifier << "\" not found" << std::endl;
  GL_CHECK(glUniform3fv(id, 1, glm::value_ptr(value)));
}

void GLShader::updateUniformMat4f(const std::string& identifier, const glm::fmat4& mat)
{
  GLint id;
  GL_CHECK(id = glGetUniformLocation(program, identifier.c_str()));
  if (id == -1)
    std::cerr << "Identifier \"" << identifier << "\" not found" << std::endl;
  GL_CHECK(glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(mat)));
}

void GLShader::updateUniformMat3f(const std::string& identifier, const glm::fmat3& mat)
{
  GLint id;
  GL_CHECK(id = glGetUniformLocation(program, identifier.c_str()));
  if (id == -1)
    std::cerr << "Identifier \"" << identifier << "\" not found" << std::endl;
  GL_CHECK(glUniformMatrix3fv(id, 1, GL_FALSE, glm::value_ptr(mat)));
}

void GLShader::updateUniformMat2f(const std::string& identifier, const glm::fmat2& mat)
{
  GLint id;
  GL_CHECK(id = glGetUniformLocation(program, identifier.c_str()));
  if (id == -1)
    std::cerr << "Identifier \"" << identifier << "\" not found" << std::endl;
  GL_CHECK(glUniformMatrix2fv(id, 1, GL_FALSE, glm::value_ptr(mat)));
}

void GLShader::updateUniform1i(const std::string& identifier, const int value)
{
   GLint id;
   GL_CHECK(id = glGetUniformLocation(program, identifier.c_str()));
   if (id == -1)
     std::cerr << "Identifier \"" << identifier << "\" not found" << std::endl;
   GL_CHECK(glUniform1i(id, value));
}

GLint GLShader::getAttribLocation(const std::string& identifier)
{
  GLint loc;
  GL_CHECK(loc = glGetAttribLocation(program, identifier.c_str()));

  if (loc == -1)
    std::cerr << "Attribute \"" << identifier << "\" not found" << std::endl;

  return loc;


}
