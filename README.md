
This is a computer graphics project I set up. The intent is to learn computer graphics concepts and brush up my C++ and CUDA skills.

### Requirements:

Have a look at the CMakeLists.txt but in short:

- CMake 3.10
- Ubuntu 16.04 or similar
- libglfw3-dev
- libglew3-dev
- libfreetype6-dev
- libgtk3-dev
- Cuda toolkit
- Recent Cuda capable GPU.

Compile and run!

Select a .obj file to open by pressing O-key. Hit Enter to switch between CUDA raytracer and OpenGL renderer. Space places an area light looking in the camera's direction.

### Planned improvements:
- Bounding volume hierarchy on the GPU
- Refractions
- Textures
- Optimization
- Make project less dependent pre-installed libraries

![Screenshot](screenshot.png?raw=true "Screenshot")
