#include "App.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <exception>
#include <sstream>

#include <glm/gtx/string_cast.hpp>

#include "nfd.h"

App::App() :
    mousePressed(false),
    mousePrevPos(glcontext.getCursorPos()),
    activeRenderer(ActiveRenderer::GL),
    glcontext(),
#ifdef ENABLE_CUDA
    cudaRenderer(),
#endif
    glmodel(),
    gllight(),
    glcanvas(glm::ivec2(WWIDTH, WHEIGHT)),
    camera(),
    loader()
{

}

App::~App()
{

}

void App::resizeCallbackEvent(int width, int height)
{
  int newWidth = width;
  int newHeight = height;

  const glm::ivec2 newSize = glm::ivec2(newWidth, newHeight);

  glcontext.resize(newSize);
  glcanvas.resize(newSize);
#ifdef ENABLE_CUDA
  cudaRenderer.resize(newSize);
#endif
}

void App::MainLoop()
{
  std::string fps = glcontext.getFPS();
  float lastUpdated = 0;

  while (glcontext.isAlive())
  {
    glcontext.clear();
    float dTime = glcontext.getDTime();
    handleControl(dTime);

    switch (activeRenderer)
    {
    case ActiveRenderer::GL:
      glcontext.draw(glmodel, gllight, camera);
      break;
#ifdef ENABLE_CUDA
      case ActiveRenderer::CUDA: // Draw image to OpenGL texture and draw with opengl
      cudaRenderer.renderToCanvas(glcanvas, camera, glmodel, gllight);
      glcontext.draw(glcanvas);
      break;
#endif
    }

    float cTime = glcontext.getTime();
    if (cTime - lastUpdated > 0.5f)
    {
      fps = glcontext.getFPS();
      glcontext.updateFPS(dTime);
      lastUpdated = cTime;
    }

    glcontext.renderText(fps + " FPS", -1.f, 0.8f);

    glcontext.swapBuffers();
  }

  createSceneFile(LAST_SCENEFILE_NAME);
}

void App::showWindow()
{
  glcontext.showWindow();
}

void App::handleControl(float dTime)
{
  // For mouse
  glm::vec2 mousePos = glcontext.getCursorPos();

  if (mousePressed)
  {
    glm::fvec2 dir = glm::fvec2(mousePos.x - mousePrevPos.x, mousePos.y - mousePrevPos.y);

    camera.rotate(dir, dTime);
  }

  mousePrevPos = mousePos;

  if (glcontext.isKeyPressed(GLFW_KEY_W))
    camera.translate(glm::vec2(0.f, 1.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_S))
    camera.translate(glm::vec2(0.f, -1.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_A))
    camera.translate(glm::vec2(1.f, 0.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_D))
    camera.translate(glm::vec2(-1.f, 0.f), dTime);
  
  if (glcontext.isKeyPressed(GLFW_KEY_RIGHT))
    camera.rotate(glm::vec2(1.f, 0.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_LEFT))
    camera.rotate(glm::vec2(-1.f, 0.f), dTime);
    
  if (glcontext.isKeyPressed(GLFW_KEY_UP))
    camera.rotate(glm::vec2(0.f, -1.f), dTime);
    
  if (glcontext.isKeyPressed(GLFW_KEY_DOWN))
    camera.rotate(glm::vec2(0.f, 1.f), dTime);
}

void App::mouseCallback(int button, int action, int /*modifiers*/)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT)
  {
    if (action == GLFW_PRESS)
      mousePressed = true;
    else if (action == GLFW_RELEASE)
      mousePressed = false;
  }
}

void App::scrollCallback(double /*xOffset*/, double yOffset)
{

  if (yOffset < 0)
    camera.increaseFOV();
  else if (yOffset > 0)
    camera.decreaseFOV();
}

void App::keyboardCallback(int key, int /*scancode*/, int action, int modifiers)
{

  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
  {
    addLight();
  }
  else if (key == GLFW_KEY_O && action == GLFW_PRESS && (modifiers & GLFW_MOD_CONTROL))
  {
    std::cout << "Choose scene file" << std::endl;
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      loadSceneFile(outPath);
      free(outPath);
    }
  }else if (key == GLFW_KEY_C && action == GLFW_PRESS)
  {
    std::cout << "Choose model file" << std::endl;
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_SaveDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      createSceneFile(outPath);
      free(outPath);
    }
  }
  else if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
  {
    activeRenderer = static_cast<App::ActiveRenderer>((activeRenderer + 1) % 2);
  }
  else if (key == GLFW_KEY_O && action == GLFW_PRESS)
  {
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      std::cout << "Opening model: " << outPath << std::endl;
      loadModel(outPath);
      free(outPath);
    }
  }
  else if (key == GLFW_KEY_L && action == GLFW_PRESS)
  {
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      std::cout << "Loading scene file: " << outPath << std::endl;
      loadSceneFile(outPath);
      free(outPath);
    }
  }

}

void App::addLight()
{
  const glm::mat4 v = camera.getView();
  const glm::mat4 l = glm::inverse(v);

  Light light(l);

  gllight.load(light);
}

void App::createSceneFile(const std::string& filename)
{
  std::ofstream sceneFile;
  sceneFile.open(filename, std::ofstream::out | std::ofstream::trunc);

  /* Order:
   *  Model filename
   *  light
   *  camera
   */

  if (!sceneFile.is_open())
  {
    std::cerr << "Couldn't write scenefile" << std::endl;
    return;
  }

  std::string modelName = glmodel.getFileName();
  sceneFile << modelName << std::endl;

  sceneFile << gllight.getLight() << std::endl;
  sceneFile << camera << std::endl;

  sceneFile.close();

  std::cout << "Wrote scene file " << filename << std::endl;
}

void App::loadModel(const std::string& modelFile)
{
  Model scene = loader.loadOBJ(modelFile);
  glmodel.load(scene);
}

void App::loadSceneFile(const std::string& filename)
{
  std::ifstream sceneFile;
  sceneFile.open(filename);

  /* Order:
   *  Model filename
   *  light
   *  camera
   */

  if (!sceneFile.is_open())
  {
    std::cerr << "Couldn't open scenefile" << std::endl;
    return;
  }

  std::string modelName;
  std::getline(sceneFile, modelName);
  loadModel(modelName);

  Light newLight;
  sceneFile >> newLight;
  gllight.load(newLight);

  sceneFile >> camera;
  sceneFile.close();

  std::cout << "Loaded scene file " << filename << std::endl;
}

void App::renderToFile(const std::string& sceneFile, const std::string& /*outfile*/)
{
  loadSceneFile(sceneFile);

#ifdef ENABLE_CUDA
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaRenderer.renderToCanvas(glcanvas, camera, glmodel, gllight);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);

  std::cout << "Rendering time [ms]: " << millis << std::endl;
#endif

  return;
}
