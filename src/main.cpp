#include <iostream>
#include <exception>

#include "cxxopts.hpp"

#include "App.hpp"

int main(int argc, char * argv[]) {

  bool batch_render = false;

  cxxopts::Options options(argv[0], "");

  options.add_options()
    ("b,batch",     "Batch render",         cxxopts::value<bool>(batch_render))
    ("r,renderer",  "Renderer type",        cxxopts::value<std::string>())
    ("p,paths",     "Number of paths",      cxxopts::value<int>())
    ("s,scene",     "Scene file",           cxxopts::value<std::string>(),  "FILE")
    ("o,output",    "Output file",          cxxopts::value<std::string>(),  "FILE");



    auto optres = options.parse(argc, argv);


    if (batch_render)
    {
#ifdef ENABLE_CUDA
      if (!optres.count("renderer"))
      {
        std::cerr << "No renderer specified" << std::endl;
        return 1;
      }

      if (!optres.count("scene"))
      {
        std::cerr << "No scene file specified" << std::endl;
        return 1;
      }

      if (!optres.count("output"))
      {
        std::cerr << "No output file specified" << std::endl;
        return 1;
      }



      std::string scenefile = optres["scene"].as<std::string>();
      std::string output = optres["output"].as<std::string>();
      std::string renderer = optres["renderer"].as<std::string>();
      int paths = 0;

      if (renderer == "pathtrace")
      {
        if (!optres.count("paths"))
        {
          std::cerr << "Number of paths not specified" << std::endl;
          return 1;
        }else
          paths = optres["paths"].as<int>();
      }

      try
      {
        App& app = App::getInstance();

        if (renderer == "raytrace")
        {
          app.rayTraceToFile(scenefile, output);
        }
        else if (renderer == "pathtrace")
        {
          app.pathTraceToFile(scenefile, output, paths);
        }else
          std::cout << "Unknown renderer" << std::endl;

      }
      catch (std::exception& e)
      {
        std::cout << e.what() << std::endl;
      }
#else
    std::cerr << "Compiled without CUDA support, no batch rendering possible. Exiting..." << std::endl;
    return EXIT_FAILURE;
#endif
  }else{

    try
    {
      App& app = App::getInstance();

      if (fileExists(LAST_SCENEFILE_NAME))
        app.loadSceneFile(LAST_SCENEFILE_NAME);

      app.showWindow();
      app.MainLoop();
    }
    catch (std::exception& e)
    {
      std::cout << e.what() << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
