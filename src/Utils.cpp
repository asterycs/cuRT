#include "Utils.hpp"
#include "Model.hpp"

#include <fstream>

#include <GL/glew.h>
#include <GL/gl.h>

#define ILUT_USE_OPENGL
#include <IL/ilu.h>

#include <glm/gtx/component_wise.hpp>

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif


void CheckOpenGLError(const char* call, const char* fname, int line)
{
  GLenum error = glGetError();

  if (error != GL_NO_ERROR)
  {
    std::string errorStr;
    switch (error)
    {
      case GL_INVALID_ENUM:                   errorStr = "GL_INVALID_ENUM"; break;
      case GL_INVALID_VALUE:                  errorStr = "GL_INVALID_VALUE"; break;
      case GL_INVALID_OPERATION:              errorStr = "GL_INVALID_OPERATION"; break;
      case GL_STACK_OVERFLOW:                 errorStr = "GL_STACK_OVERFLOW"; break;
      case GL_STACK_UNDERFLOW:                errorStr = "GL_STACK_UNDERFLOW"; break;
      case GL_OUT_OF_MEMORY:                  errorStr = "GL_OUT_OF_MEMORY"; break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:  errorStr = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
      default:                                errorStr = "Unknown error"; break;
    }

    std::cerr << "At: " << fname << ":" << line << std::endl \
     << " OpenGL call: " << call << std::endl \
      << " Error: " << errorStr << std::endl;
  }
}

void CheckILError(const char* call, const char* fname, int line)
{
  ILenum error = ilGetError();

  if (error != IL_NO_ERROR) {
    do {
      std::string errStr = iluErrorString(error);

      std::cerr << "At: " << fname << ":" << line << std::endl \
       << " IL call: " << call << std::endl \
        << " Error: " << errStr << std::endl;

    } while ((error = ilGetError ()));
  }
}

#ifdef ENABLE_CUDA
void CheckCudaError(const char* call, const char* fname, int line)
{
    cudaError_t result_ = cudaGetLastError();
    if (result_ != cudaSuccess) {
        std::cerr << "At: " << fname << ":" << line << std::endl \
           << " Cuda call: " << call << " Error: " << cudaGetErrorString(result_) << std::endl;
        exit(1);
    }
}
#endif

std::string readFile(const std::string& filePath) {
    std::string content;
    std::ifstream fileStream(filePath, std::ios::in);

    if(!fileStream.is_open()) {
        std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
        return "";
    }

    std::string line = "";
    while(!fileStream.eof()) {
        std::getline(fileStream, line);
        content.append(line + "\n");
    }

    fileStream.close();
    return content;
}

bool fileExists(const std::string& filename)
{
    std::ifstream infile(filename);
    return infile.good();
}

CUDA_FUNCTION float AABB::area() const
{
  glm::fvec3 d = max - min;

  return 2 * (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
}

CUDA_FUNCTION unsigned int AABB::maxAxis() const
{
  const glm::fvec3 d = glm::abs(max - min);
  const float mv = glm::compMax(d);

  for (unsigned int i = 0; i < 3; ++i)
  {
    if (d[i] == mv)
      return i;
  }

  return 0;
}

CUDA_FUNCTION void AABB::add(const Triangle& t)
{
  for (auto& v : t.vertices)
    add(v.p);
}

CUDA_FUNCTION void AABB::add(const glm::fvec3 v)
{
  min = glm::min(min, v);
  max = glm::max(max, v);
}
