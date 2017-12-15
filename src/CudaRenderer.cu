#include "CudaRenderer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include <glm/gtx/component_wise.hpp>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Utils.hpp"
#include "Triangle.hpp"

#define BLOCKWIDTH 8
#define EPSILON 0.0001f
#define BIGT 99999.f
#define SHADOWSAMPLING 64
#define REFLECTIONS 1

__device__ bool bboxIntersect(const AABB& box, const Ray& ray, float& t)
{
  glm::fvec3 tmin(-BIGT), tmax(BIGT);

  glm::fvec3 tdmin = (box.min - ray.origin) * ray.inverseDirection;
  glm::fvec3 tdmax = (box.max - ray.origin) * ray.inverseDirection;

  tmin = glm::min(tdmin, tdmax);
  tmax = glm::max(tdmin, tdmax);

  float tmind = glm::compMax(tmin);
  float tmaxd = glm::compMin(tmax);

  t = min(tmind, tmaxd);

  return tmaxd >= tmind && !(tmaxd < 0.f && tmind < 0.f);
}

__device__ void debug_vec3f(const glm::fvec3& v)
{
  printf("%f %f %f\n", v.x, v.y, v.z);
}

__device__ glm::fvec3 mirrorDirection(const glm::vec3& normal, const glm::vec3& incoming) {
  glm::vec3 ret = incoming - 2 * glm::dot(incoming, normal) * normal;
  return glm::normalize(ret);
}

__device__ bool rayTriangleIntersection(const Ray& ray, const Triangle& triangle, float& t)
{
  /* Möller-Trumbore algorithm
   * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
   */

  const glm::vec3& vertex0 = triangle.vertices[0].p;
  const glm::vec3& vertex1 = triangle.vertices[1].p;
  const glm::vec3& vertex2 = triangle.vertices[2].p;
  glm::vec3 edge1, edge2, h, s, q;
  float a,f,u,v;
  edge1 = vertex1 - vertex0;
  edge2 = vertex2 - vertex0;

  h = glm::cross(ray.direction, edge2);
  a = glm::dot(edge1, h);

  if (a > -EPSILON && a < EPSILON)
    return false;

  f = 1 / a;
  s = ray.origin - vertex0;
  u = f * glm::dot(s, h);

  if (u < 0.f || u > 1.0f)
    return false;

  q = glm::cross(s, edge1);
  v = f * glm::dot(ray.direction, q);

  if (v < 0.0 || u + v > 1.0)
    return false;

  t = f * glm::dot(edge2, q);

  if (t > EPSILON)
  {
    return true;
  }
  else
    return false;
}


__device__ RaycastResult rayCast(const Ray& ray, const Node* bvh, const Triangle* triangles, const unsigned int nTriangles)
{
  float tMin = BIGT;
  int minTriIdx = -1;
  RaycastResult result;

  float hitt = BIGT;
  bool hit = bboxIntersect(bvh[0].bbox, ray, hitt);

  if (!hit)
    return result;

  int ptr = 1;
  int stack[16] { 0 };

  while (ptr > 0)
  {
    --ptr;

    int currentNodeIdx = stack[ptr];
    Node currentNode = bvh[currentNodeIdx];


    if (currentNode.rightIndex == -1)
    {
      float t = 0;

      for (int i = currentNode.startTri; i < currentNode.startTri + currentNode.nTri; ++i)
      {
        if (rayTriangleIntersection(ray, triangles[i], t))
        {
          if(t < tMin)
          {
            tMin = t;
            minTriIdx = i;
          }
        }
      }

    }else
    {
      Node leftChild = bvh[stack[ptr] + 1];
      Node rightChild = bvh[currentNode.rightIndex];

      float leftt, rightt;
      bool leftHit = bboxIntersect(leftChild.bbox, ray, leftt);
      bool rightHit = bboxIntersect(rightChild.bbox, ray, rightt);

      if (leftHit)
      {
        stack[ptr] = currentNodeIdx + 1;
        ++ptr;
      }

      if (rightHit)
      {
        stack[ptr] = currentNode.rightIndex;
        ++ptr;
      }
    }

  }

  if (minTriIdx == -1)
    return result;

  glm::fvec3 hitPoint = ray.origin + glm::normalize(ray.direction) * tMin;
  glm::fvec2 uv(0.f);

  result.point = hitPoint;
  result.t = tMin;
  result.triangleIdx = minTriIdx;
  result.uv = uv;

  return result;
}


template<typename curandState>
__device__ glm::fvec3 areaLightShading(const Light& light, const Node* bvh, const RaycastResult& result, const Triangle* triangles, const unsigned int nTriangles, curandState& curandState1, curandState& curandState2, const unsigned int supersampling)
{
  glm::fvec3 brightness(0.f);

  if (!light.isEnabled())
    return brightness;

  const Triangle& hitTriangle = triangles[result.triangleIdx];

  glm::fvec3 lightSamplePoint;
  float pdf;

  for (unsigned int i = 0; i < supersampling; ++i) // Maybe utilize dynamic parallelism here?
  {
    light.sample(pdf, lightSamplePoint, curandState1, curandState2);

    glm::fvec3 shadowRayOrigin = result.point + hitTriangle.normal() * EPSILON;
    glm::fvec3 shadowRayDir = lightSamplePoint - shadowRayOrigin;

    float maxT = glm::length(shadowRayDir); // Distance to the light

    shadowRayDir = glm::normalize(shadowRayDir);

    Ray shadowRay(shadowRayOrigin, glm::normalize(shadowRayDir));

    RaycastResult shadowResult = rayCast(shadowRay, bvh, triangles, nTriangles);

    if ((shadowResult && shadowResult.t >= maxT - EPSILON) || !shadowResult)
    {

      const float cosOmega = glm::clamp(glm::dot(shadowRayDir, hitTriangle.normal()), 0.f, 1.f);
      const float cosL = glm::clamp(glm::dot(-shadowRayDir, light.getNormal()), 0.f, 1.f);

      brightness += 1.0f / (glm::dot(shadowRayDir, shadowRayDir) * pdf) * light.getEmission() * cosOmega * cosL;

    }
  }

  brightness /= supersampling;

  return brightness;
}

__device__ glm::fvec3 rayTrace(\
    const Node* bvh, \
    const Ray& ray, \
    const Triangle* triangles, \
    const int nTriangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandState_t& curandState1, \
    curandState_t& curandState2, \
    const int reflections)
{

  int it = reflections;
  glm::fvec3 color(0.f);

  // Primary
  RaycastResult result = rayCast(ray, bvh, triangles, nTriangles);

  if (!result)
    return color;

  const Triangle* hitTriangle = &triangles[result.triangleIdx];
  const Material* material = &materials[triangleMaterialIds[result.triangleIdx]];

  color = material->colorAmbient * 0.25f; // Ambient lightning
  color += material->colorDiffuse / glm::pi<float>() * areaLightShading(light, bvh, result, triangles, nTriangles, curandState1, curandState2, SHADOWSAMPLING);

  glm::fvec3 filterSpecular = material->colorSpecular;
  glm::fvec3 reflRayOrigin;
  glm::fvec3 reflRayDir;

  // Secondary or reflections
  while (glm::length(filterSpecular) > 0.0f && it)
  {
    reflRayOrigin = result.point + hitTriangle->normal() * EPSILON;
    reflRayDir = mirrorDirection(hitTriangle->normal(), ray.direction);
    Ray reflRay = Ray(reflRayOrigin, reflRayDir);
    result = rayCast(reflRay, bvh, triangles, nTriangles);

    if (!result)
      break;

    hitTriangle = &triangles[result.triangleIdx];
    material = &materials[triangleMaterialIds[result.triangleIdx]];

    color += filterSpecular * material->colorAmbient * 0.25f; // Ambient lightning
    color += filterSpecular * material->colorDiffuse / glm::pi<float>() * areaLightShading(light, bvh, result, triangles, nTriangles, curandState1, curandState2, SHADOWSAMPLING);

    filterSpecular = material->colorSpecular;

    --it;
  }

  return color;
}

template<typename curandState>
__global__ void initRand(const int /*seed*/, curandState* const curandStateDevPtr, const glm::ivec2 size)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= size.x || y >= size.y)
    return;

  curandState localState;
  curand_init(x + y*size.x, 0, 0, &localState);
  curandStateDevPtr[x + y * size.x] = localState;
}

template<typename T>
__device__ void writeToCanvas(const unsigned int x, const unsigned int y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2& canvasSize, T& data)
{
  float4 out = make_float4(data.x, data.y, data.z, 1.f);
  surf2Dwrite(out, surfaceObj, (canvasSize.x - 1 - x) * sizeof(out), y);
  return;
}

__global__ void cudaRender(\
    const cudaSurfaceObject_t canvas, \
    const glm::ivec2 canvasSize, \
    const Triangle* triangles, \
    const int nTriangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandState_t* curandStateDevXPtr, \
    curandState_t* curandStateDevYPtr, \
    const Node* bvh)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  glm::vec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(glm::ivec2(x, y), canvasSize);

  const float ar = (float) canvasSize.x/canvasSize.y;

  Ray ray = camera.generateRay(nic, ar);

  glm::fvec3 color = rayTrace(\
      bvh,
      ray, \
      triangles, \
      nTriangles, \
      camera, \
      materials, \
      triangleMaterialIds, \
      light, \
      curandStateDevXPtr[x + canvasSize.x * y], \
      curandStateDevYPtr[x + canvasSize.x * y], \
      REFLECTIONS);

  writeToCanvas(x, y, canvas, canvasSize, color);

  return;
}

void CudaRenderer::resize(const glm::ivec2& size)
{
  curandStateDevVecX.resize(size.x * size.y);
  curandStateDevVecY.resize(size.x * size.y);
  curandState_t* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  curandState_t* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid( (size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);
  initRand<<<grid, block>>>(0, curandStateDevXRaw, size);
  initRand<<<grid, block>>>(5, curandStateDevYRaw, size);
  CUDA_CHECK(cudaDeviceSynchronize()); // Would need to wait anyway when initializing models etc.
}


CudaRenderer::CudaRenderer() : curandStateDevVecX(), curandStateDevVecY()
{
  unsigned int cudaDeviceCount = 0;
  int cudaDevices[8];
  unsigned int cudaDevicesCount = 8;

	cudaGLGetDevices(&cudaDeviceCount, cudaDevices, cudaDevicesCount, cudaGLDeviceListCurrentFrame);

  if (cudaDeviceCount < 1)
  {
     std::cout << "No CUDA devices found" << std::endl;
     throw std::runtime_error("No CUDA devices available");
  }

  CUDA_CHECK(cudaSetDevice(cudaDevices[0]));

  resize(glm::ivec2(WWIDTH, WHEIGHT));
}

CudaRenderer::~CudaRenderer()
{

}

void CudaRenderer::renderToCanvas(GLCanvas& canvas, const Camera& camera, GLModel& model, GLLight& light)
{
  if (model.getNTriangles() == 0)
    return;

  const glm::ivec2& canvasSize = canvas.getSize();

  curandState_t* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  curandState_t* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  auto surfaceObj = canvas.getCudaMappedSurfaceObject();
  Triangle* devTriangles = model.getMappedCudaTrianglePtr();

  int meshes = model.getNMeshes();

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid( (canvasSize.x+ block.x - 1) / block.x, (canvasSize.y + block.y - 1) / block.y);

  cudaRender<<<grid, block>>>(\
      surfaceObj, \
      canvasSize, \
      devTriangles, \
      model.getNTriangles(), \
      camera, \
      model.getCudaMaterialsPtr(), \
      model.getCudaTriangleMaterialIdsPtr(), \
      light.getLight(), \
      curandStateDevXRaw, \
      curandStateDevYRaw, \
      model.getDeviceBVH());

  CUDA_CHECK(cudaDeviceSynchronize());


  model.unmapCudaTrianglePtr();
  canvas.cudaUnmap();
}


