#ifndef APP_HPP
#define APP_HPP

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include "ModelLoader.hpp"
#include "GLContext.hpp"
#include "GLCanvas.hpp"

#ifdef ENABLE_CUDA
  #include "CudaRenderer.hpp"
#endif

#define LAST_SCENEFILE_NAME "last.scene"

class App { 
public:
    // Singleton
    App(App const&) = delete;
    void operator=(App& app) = delete;
    static App& getInstance() {static App app; return app;}

    void showWindow();

    void MainLoop();

    void mouseCallback(int button, int action, int modifiers);
    void scrollCallback(double xOffset, double yOffset);
    void keyboardCallback(int key, int scancode, int action, int modifiers);
    void resizeCallbackEvent(int width, int height);
    void initProjection(int width, int height, float near, float far);
    void handleControl(float dTime);
    void addLight();
    void createSceneFile(const std::string& filename);
    void loadSceneFile(const std::string& filename);
    void loadModel(const std::string& modelFile);

    void renderToFile(const std::string& sceneFile, const std::string& outFile);
private:
    enum ActiveRenderer {
      GL
#ifdef ENABLE_CUDA
      , CUDA
#endif
    };
    
    App();
    ~App();

    std::vector<glm::fvec3> debugPoints;
    bool mousePressed;
    glm::dvec2 mousePrevPos;
    ActiveRenderer activeRenderer;

    GLContext glcontext;
#ifdef ENABLE_CUDA
    CudaRenderer cudaRenderer;
#endif
    Model model;
    GLModel glmodel;
    GLLight gllight;
    GLCanvas glcanvas;
    
    Camera camera;
    ModelLoader loader;

    bool drawDebug;
    unsigned int debugBboxPtr;
};

#endif
