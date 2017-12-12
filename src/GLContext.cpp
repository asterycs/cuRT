#include "GLContext.hpp"
#include "App.hpp"
#include "Utils.hpp"

#include <glm/gtx/string_cast.hpp>

GLContext::GLContext() :
  lastTime(static_cast<float>(glfwGetTime())),
  fps(0.f),
  modelShader(),
  lightShader(),
  canvasShader(),
  window(nullptr),
  size(WWIDTH, WHEIGHT),
  ftOperational(false)
{
  if (!glfwInit())
  {
    std::cerr << "GLFW init failed" << std::endl;
  }

  glfwSetTime(0);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_SAMPLES, 0);
  glfwWindowHint(GLFW_RED_BITS, 8);
  glfwWindowHint(GLFW_GREEN_BITS, 8);
  glfwWindowHint(GLFW_BLUE_BITS, 8);
  glfwWindowHint(GLFW_ALPHA_BITS, 8);
  glfwWindowHint(GLFW_STENCIL_BITS, 8);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
  glfwWindowHint(GLFW_SAMPLES, 8); // hehe

  window = glfwCreateWindow(size.x, size.y, "GRAPHX", nullptr, nullptr);
  if (window == nullptr) {
      throw std::runtime_error("Failed to create GLFW window");
  }
  GL_CHECK(glfwMakeContextCurrent(window));

  glewExperimental = GL_TRUE;
  GLenum err = glewInit();
  if(err!=GLEW_OK) {
    throw std::runtime_error("glewInit failed");
  }

  // Defuse bogus error
  glGetError();

  GL_CHECK(glClearColor(0.2f, 0.25f, 0.3f, 1.0f));
  GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  int width, height;
  GL_CHECK(glfwGetFramebufferSize(window, &width, &height));
  GL_CHECK(glViewport(0, 0, width, height));
  glfwSwapInterval(0);
  glfwSwapBuffers(window);

  glfwSetMouseButtonCallback(window,
      [](GLFWwindow *, int button, int action, int modifiers) {
          App::getInstance().mouseCallback(button, action, modifiers);
      }
  );
  
  glfwSetScrollCallback(window,
      [](GLFWwindow *, double xOffset, double yOffset) {
          App::getInstance().scrollCallback(xOffset, yOffset);
      }
  );

  glfwSetKeyCallback(window,
      [](GLFWwindow *, int key, int scancode, int action, int mods) {
          App::getInstance().keyboardCallback(key, scancode, action, mods);
      }
  );

  glfwSetWindowSizeCallback(window,
      [](GLFWwindow *, int width, int height) {
          App::getInstance().resizeCallbackEvent(width, height);
      }
  );

  glfwSetErrorCallback([](int error, const char* description) {
    std::cout << "Error: " << error << " " << description << std::endl;
  });

  modelShader.loadShader("shaders/model/vshader.glsl", "shaders/model/fshader.glsl");
  lightShader.loadShader("shaders/light/vshader.glsl", "shaders/light/fshader.glsl");
  canvasShader.loadShader("shaders/canvas/vshader.glsl", "shaders/canvas/fshader.glsl");
  textShader.loadShader("shaders/text/vshader.glsl", "shaders/text/fshader.glsl");
  depthShader.loadShader("shaders/depth/vshader.glsl", "shaders/depth/fshader.glsl");
  
  std::cout << "OpenGL context initialized" << std::endl;

  ftOperational = initFT();
}

bool GLContext::initFT()
{
  if(FT_Init_FreeType(&ft)) {
    std::cerr << "Could not init freetype library" << std::endl;
    return false;
  }

  if(FT_New_Face(ft, "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf", 0, &face)) {
    std::cerr << "Could not open font" << std::endl;
    return false;
  }

  FT_Set_Pixel_Sizes(face, 0, 48);

  return true;
}

void GLContext::renderText(const std::string& text, const float x, const float y) {

  if (!ftOperational)
    return;

  textShader.bind();

  FT_GlyphSlot g = face->glyph;
  const float sx = 2.f / size.x;
  const float sy = 2.f / size.y;

  GLuint tex;
  GL_CHECK(glActiveTexture(GL_TEXTURE0));
  GL_CHECK(glGenTextures(1, &tex));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, tex));
  textShader.updateUniform1i("texture", 0);

  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));

  GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));

  GLint posLoc = textShader.getAttribLocation("position");

  GLuint vbo, vao;
  GL_CHECK(glGenVertexArrays(1, &vao));
  GL_CHECK(glBindVertexArray(vao));
  GL_CHECK(glGenBuffers(1, &vbo));
  GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vbo));
  GL_CHECK(glVertexAttribPointer(posLoc, sizeof(GLfloat), GL_FLOAT, GL_FALSE, 0, NULL));
  GL_CHECK(glEnableVertexAttribArray(posLoc));

  GL_CHECK(glEnable(GL_BLEND));
  GL_CHECK(glDisable(GL_DEPTH_TEST));
  GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));



  float posX = x;
  float posY = y;

  for(auto c : text) {
    if(FT_Load_Char(face, c, FT_LOAD_RENDER))
        continue;

    GL_CHECK(glTexImage2D(
      GL_TEXTURE_2D,
      0,
      GL_RED,
      g->bitmap.width,
      g->bitmap.rows,
      0,
      GL_RED,
      GL_UNSIGNED_BYTE,
      g->bitmap.buffer
    ));

    const float x2 = posX + g->bitmap_left * sx;
    const float y2 = -posY - g->bitmap_top * sy;
    const float w = g->bitmap.width * sx;
    const float h = g->bitmap.rows * sy;

    const GLfloat aabb[4][4] = {
        {x2,     -y2    , 0.f, 0.f},
        {x2 + w, -y2    , 1.f, 0.f},
        {x2,     -y2 - h, 0.f, 1.f},
        {x2 + w, -y2 - h, 1.f, 1.f},
    };

    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(aabb), aabb, GL_STATIC_DRAW));
    GL_CHECK(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));

    posX += (g->advance.x/64) * sx;
    posY += (g->advance.y/64) * sy;
  }

  GL_CHECK(glEnable(GL_DEPTH_TEST));
  GL_CHECK(glDisable(GL_BLEND));
  GL_CHECK(glDisableVertexAttribArray(posLoc));
  GL_CHECK(glDeleteTextures(1, &tex));
  GL_CHECK(glDeleteBuffers(1, &vbo));
  GL_CHECK(glDeleteVertexArrays(1, &vao));

  textShader.unbind();
}

GLContext::~GLContext()
{
  glfwDestroyWindow(window);
  glfwTerminate();
}

bool GLContext::shadersLoaded() const
{
  if (modelShader.isLoaded() \
 && lightShader.isLoaded() \
 && canvasShader.isLoaded() \
 && depthShader.isLoaded() \
 && textShader.isLoaded())
    return true;
  else
    return false;
}

void GLContext::clear()
{
  if (!shadersLoaded())
    return;
    
  GL_CHECK(glClearColor(0.2f, 0.25f, 0.3f, 1.0f));
  GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  GL_CHECK(glEnable(GL_DEPTH_TEST));

  glfwPollEvents();
}

void GLContext::swapBuffers()
{
  glfwSwapBuffers(window);
}

bool GLContext::isAlive()
{
  return !glfwWindowShouldClose(window);
}

float GLContext::getDTime()
{
  float currentTime = getTime();
  float dTime = currentTime - lastTime;
  lastTime = currentTime;

  return dTime;
}

float GLContext::getTime() const
{
  return (float) glfwGetTime();
}

std::string GLContext::getFPS()
{
  std::string s(std::to_string(std::floor(fps)));

  // Must look for both decimal separators. File dialogue changes sprintf behavior according to locale...
  while( ((s.find_first_of(",.") != std::string::npos) && (s.substr(s.length() - 1, 1) == "0")) || (s.find_first_of(",.") == s.length() - 1))
  {
    s.pop_back();
  }

  return s;
}

void GLContext::draw(const GLModel& model, const GLLight& light, const Camera& camera)
{
  updateShadowMap(model, light);
  //drawShadowMap(light); // For debug
  drawModel(model, camera, light);
  drawLight(light, camera);
}

void GLContext::updateFPS(const float dTime)
{
  fps = 0.5f * fps + 0.5 * 1/dTime;
}

void GLContext::drawShadowMap(const GLLight& light)
{
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  const glm::ivec2 shadowMapSize = light.getShadowMap().getSize();
  glViewport(0,0,shadowMapSize.x, shadowMapSize.y);
  
  GL_CHECK(glActiveTexture(GL_TEXTURE0));
  canvasShader.bind();
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, light.getShadowMap().getDepthTextureID()));
  GL_CHECK(glBindVertexArray(light.getShadowMap().getVaoID()));

  canvasShader.updateUniform1i("texture", 0);

  GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 3));

  GL_CHECK(glBindVertexArray(0));
  canvasShader.unbind();
}

void GLContext::updateShadowMap(const GLDrawable& model, const GLLight& light)
{
  GLuint frameBufferID = light.getShadowMap().getFrameBufferID();

  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID));
  glClear(GL_DEPTH_BUFFER_BIT);
  GL_CHECK(glEnable(GL_DEPTH_TEST));
  GL_CHECK(glCullFace(GL_FRONT));

  const auto vaoID = model.getVaoID();
  const auto& meshDescriptors = model.getMeshDescriptors();
  const glm::ivec2 depthTextureSize = light.getShadowMap().getSize();
  GL_CHECK(glViewport(0, 0, depthTextureSize.x, depthTextureSize.y));

  depthShader.bind();
  GL_CHECK(glBindVertexArray(vaoID));
  GL_CHECK(depthShader.updateUniformMat4f("MVP", light.getDepthMVP()));

  for (auto& meshDescriptor : meshDescriptors)
  {
    GL_CHECK(glDrawElements(GL_TRIANGLES, meshDescriptor.vertexIds.size(), GL_UNSIGNED_INT, meshDescriptor.vertexIds.data()));
  }

  depthShader.unbind();
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
  GL_CHECK(glBindVertexArray(0));
  return;
}

void GLContext::drawModel(const GLModel& model, const Camera& camera, const GLLight& light)
{
  if (!shadersLoaded() || model.getNTriangles() == 0)
    return;

  GL_CHECK(glViewport(0,0,size.x, size.y));
  GL_CHECK(glCullFace(GL_BACK));
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));

  const auto& vaoID = model.getVaoID();
  const auto& meshDescriptors = model.getMeshDescriptors(); // TODO: Switch between bvh visualization and normal colors
  const auto& materials = model.getMaterials();
  //const auto& meshDescriptors = model.getBVHBoxDescriptors();

  modelShader.bind();
  modelShader.updateUniformMat4f("posToCamera", camera.getMVP(size));
  //modelShader.updateUniformMat3f("normalToCamera", glm::mat3(glm::transpose(glm::inverse(camera.getMVP(size)))));
  modelShader.updateUniformMat4f("biasedDepthToLight", light.getDepthBiasMVP());
  modelShader.updateUniform3fv("lightPos", light.getLight().getPosition());
  modelShader.updateUniform3fv("lightNormal", light.getLight().getNormal());

  GL_CHECK(glActiveTexture(GL_TEXTURE0));
  modelShader.updateUniform1i("shadowMap", 0);
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, light.getShadowMap().getDepthTextureID()));
  GL_CHECK(glBindVertexArray(vaoID));

  for (auto& meshDescriptor : meshDescriptors)
  {
    auto& material = materials[meshDescriptor.materialIdx];
    
    modelShader.updateUniform3fv("material.colorAmbient", material.colorAmbient);
    modelShader.updateUniform3fv("material.colorDiffuse", material.colorDiffuse);
    //modelShader.updateUniform3fv("material.colorSpecular", material.colorSpecular);

    GL_CHECK(glDrawElements(GL_TRIANGLES, meshDescriptor.vertexIds.size(), GL_UNSIGNED_INT, meshDescriptor.vertexIds.data()));
  }

  GL_CHECK(glBindVertexArray(0));
  modelShader.unbind();
}

void GLContext::drawLight(const GLLight& light, const Camera& camera)
{
  if (!shadersLoaded() || light.getNTriangles() == 0)
    return;

  GL_CHECK(glViewport(0,0,size.x, size.y));
  GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
  lightShader.bind();

  const auto& vaoID = light.getVaoID();
  GL_CHECK(glBindVertexArray(vaoID));
  lightShader.updateUniformMat4f("MVP", camera.getMVP(size));
  
  GL_CHECK(glEnable(GL_CULL_FACE));
  GL_CHECK(glCullFace(GL_FRONT));
  lightShader.updateUniform3fv("lightColor", glm::vec3(0.f, 0.f, 0.f));
	GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, light.getNTriangles() * 3));

  GL_CHECK(glCullFace(GL_BACK));
  glm::fvec3 frontLightCol;

  if (light.getLight().isEnabled())
    frontLightCol = glm::fvec3(1.f, 1.f, 1.f);
  else
    frontLightCol = glm::fvec3(0.3f, 0.3f, 0.3f);


  lightShader.updateUniform3fv("lightColor", glm::vec3(1.f, 1.f, 1.f));
  GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, light.getNTriangles() * 3));

	GL_CHECK(glBindVertexArray(0));
	lightShader.unbind();
  GL_CHECK(glDisable(GL_CULL_FACE));
}

glm::vec2 GLContext::getCursorPos()
{
  double x, y;
  glfwGetCursorPos(window, &x, &y);

  return glm::vec2(x, y);
}

bool GLContext::isKeyPressed(const int glfwKey) const
{
  return glfwGetKey(window, glfwKey); 
}

void GLContext::resize(const glm::ivec2& newSize)
{
  GL_CHECK(glViewport(0,0,newSize.x, newSize.y));
  this->size = newSize;
}

void GLContext::draw(const GLCanvas& canvas)
{
  if (!shadersLoaded() || canvas.getTextureID() == 0)
     return;

   GL_CHECK(glActiveTexture(GL_TEXTURE0));
   canvasShader.bind();
   GL_CHECK(glBindTexture(GL_TEXTURE_2D, canvas.getTextureID()));
   GL_CHECK(glBindVertexArray(canvas.getVaoID()));
   
   canvasShader.updateUniform1i("texture", 0);
  
   GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, canvas.getNTriangles() * 3));

   GL_CHECK(glBindVertexArray(0));
   canvasShader.unbind();
}

glm::ivec2 GLContext::getSize() const
{
  return size;
}
