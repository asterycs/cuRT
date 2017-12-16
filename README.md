
This is a computer graphics project I set up. The intent is to learn computer graphics concepts and brush up my C++ and CUDA skills.

### Requirements:

Have a look at the CMakeLists.txt but in short:

- CMake 3.10
- Ubuntu 16.04 or similar
- libx11-dev
- libglew3-dev
- libfreetype6-dev
- libgtk3-dev
- Cuda toolkit
- Recent Cuda capable GPU.

Most dependencies are handled with CMake.

### Build
```
cd $DOWNLOAD_DIR
mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=TRUE ..
```

Select an .obj file to open by pressing O-key. Hit Enter to switch between CUDA raytracer and OpenGL renderer. Space places an area light looking in the camera's direction.

### Current features:
- OpenGL preview
    - Shadow maps
- Raytracer in CUDA
    - Area lights with soft shadows
    - Reflections
    - Simple BVH based on morton codes

### Planned improvements:
- Better BVH
- Refractions
- Textures

![Screenshot](screenshot.png?raw=true "Screenshot")
