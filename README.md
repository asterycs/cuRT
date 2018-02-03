
This is a computer graphics project I set up. The intent is to learn computer graphics concepts and brush up my C++ and CUDA skills.

### Requirements:

Have a look at the CMakeLists.txt but in short:

- CMake 3.10
- Ubuntu 16.04 or similar
- libx11-dev
- libglew3-dev
- libgtk3-dev
- Cuda toolkit 8.0 (9.0 seems incompatible with glm)
- Recent Cuda capable GPU.

This program additionally uses:
- Assimp for model loading
  - With a small patch that can be found under patches/
- Dear IMGUI for user interface
- Nativefiledialog
- cxxopts
- glm
- glfw3

These dependencies are handled by CMake.

### Build
```
cd $DOWNLOAD_DIR
mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=TRUE ..
```

Select an .obj file to open by pressing O-key. Hit Enter to switch between CUDA raytracer and OpenGL renderer. Space places an area light looking in the camera's direction.

The kernels have been optimized for the NVIDIA GTX 1060 3GB that I own.

### Current features:
- Simple BVH based on morton codes
- SAH based bvh
- OpenGL preview
    - Shadow maps
    - Ray visualization (ctrl + D)
    - BVH visualization
- A ray tracer and a path tracer in CUDA
    - Area lights with soft shadows and quasirandom sampling
    - Reflections
    - Refractions

### Planned improvements:
- Better BVH
- Textures
- More optimization

![Screenshot1](raytrace.png?raw=true "raytrace")
Ray tracer
![Screenshot2](pathtrace.png?raw=true "pathtrace")
Path tracer
![Screenshot3](bvh.png?raw=true "Debug visualization")
BVH Visualization

Screenshot model downloaded from Morgan McGuire's Computer Graphics Archive https://casual-effects.com/data
