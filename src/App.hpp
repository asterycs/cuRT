#ifndef APP_HPP
#define APP_HPP

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

#include "ModelLoader.hpp"
#include "GLContext.hpp"
#include "GLTexture.hpp"

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
    void drawDebugInfo();

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

#ifdef ENABLE_CUDA
    void rayTraceToFile(const std::string& sceneFile, const std::string& outFile);
    void pathTraceToFile(const std::string& sceneFile, const std::string& outFile, const int paths);
#endif
private:
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
    GLTexture glcanvas;
    
    Camera camera;
    ModelLoader loader;

    enum DebugMode debugMode;
    unsigned int debugBboxPtr;
};

#endif
