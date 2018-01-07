#include <iostream>
#include <exception>

#include "cxxopts.hpp"

#include "App.hpp"

int main(int argc, char * argv[]) {

  bool batch_render = false;

  cxxopts::Options options(argv[0], " - example command line options");
  options.positional_help("[optional args]");

  options.add_options()
    ("b,batch",     "Batch render",         cxxopts::value<bool>(batch_render))
    ("r,renderer",  "Renderer type",        cxxopts::value<int>())
    ("t,timeout",   "Path tracing timeout", cxxopts::value<float>())
    ("s,scene",     "Scene file",           cxxopts::value<std::string>(),  "FILE")
    ("o,output",    "Output file",          cxxopts::value<std::string>(),  "FILE");


    try
    {
        options.parse(argc, argv);
    }catch (const cxxopts::OptionException& e)
    {
      std::cout << "Invalid parameter: " << e.what() << std::endl;
      return 1;
    }

    if (batch_render)
    {

      if (!options.count("renderer"))
      {
        std::cerr << "No renderer specified" << std::endl;
        return 1;
      }

      if (!options.count("scene"))
      {
        std::cerr << "No scene file specified" << std::endl;
        return 1;
      }

      if (!options.count("output"))
      {
        std::cerr << "No output file specified" << std::endl;
        return 1;
      }



      std::string scenefile = options["scene"].as<std::string>();
      std::string output = options["output"].as<std::string>();
      int renderer = options["renderer"].as<int>();
      float timeout = 0;

      if (renderer == 1)
      {
        if (!options.count("timeout"))
        {
          std::cerr << "No timeout specified" << std::endl;
          return 1;
        }else
          timeout = options["timeout"].as<float>();
      }

      try
      {
        App& app = App::getInstance();

        switch (renderer)
        {
        case 0:
          app.rayTraceToFile(scenefile, output);
          break;
        case 1:
          app.pathTraceToFile(scenefile, output, timeout);
          break;

        default:
          std::cout << "Unknown renderer" << std::endl;
        }
      }
      catch (std::exception& e)
      {
        std::cout << e.what() << std::endl;
      }

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
    }
  }

  return 0;
}
