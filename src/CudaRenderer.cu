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
#define SHADOWSAMPLING 1
#define SECONDARY_RAYS 1
#define AIR_INDEX 0.8f

__device__ bool bboxIntersect(const AABB& box, const Ray& ray, float& t)
{
  glm::fvec3 tmin(-BIGT), tmax(BIGT);

  glm::fvec3 tdmin = (box.min - ray.origin) * ray.inverseDirection;
  glm::fvec3 tdmax = (box.max - ray.origin) * ray.inverseDirection;

  tmin = glm::min(tdmin, tdmax);
  tmax = glm::max(tdmin, tdmax);

  float tmind = glm::compMax(tmin);
  float tmaxd = glm::compMin(tmax);

  t = fminf(tmind, tmaxd);

  return tmaxd >= tmind && !(tmaxd < 0.f && tmind < 0.f);
}

__device__ void debug_vec3f(const glm::fvec3& v)
{
  printf("%f %f %f\n", v.x, v.y, v.z);
}

inline __device__ glm::fvec3 reflectionDirection(const glm::vec3& normal, const glm::vec3& incoming) {

  const float cosT = glm::dot(incoming, normal);

  if (cosT > 0.f)
    return incoming - 2 * cosT * -normal;
  else
    return incoming - 2 * cosT * normal;
}

inline __device__ glm::fvec3 refractionDirection(const glm::vec3& normal, const glm::vec3& incoming, const float index1, const float index2) {
  /*float sinOutAng = (sin(acos(cosInAng) * index1)) / index2;
  glm::fvec3 den = (incoming + cosInAng * normal) * sinOutAng;

  glm::fvec3 ret = den / sin(acos(cosInAng)) - normal * cos(asin(sinOutAng));

  return ret;*/

  const float cosInAng = fabsf(glm::dot(incoming, normal));
  const float sin2t = (index1 / index2) * (index1 / index2) * (1 - cosInAng * cosInAng);

  if (sin2t > 1)
    return reflectionDirection(normal, incoming);
  else
    return index1 / index2 * incoming + (index1 / index2 * cosInAng - sqrt(1 - sin2t)) * normal;
}

__device__ bool rayTriangleIntersection(const Ray& ray, const Triangle& triangle, float& t, glm::fvec2& uv)
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

  f = __fdividef(1.f, a);
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
    uv = glm::fvec2(u, v);
    return true;
  }
  else
    return false;
}

__global__ void dynamicIntersection(const Ray ray, const Triangle triangles[], const unsigned int nTris, float* minT, unsigned int* tri)
{
  const int tid = threadIdx.x;
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= nTris)
    return;

  __shared__ float ts[8];
  __shared__ unsigned int triIds[8];

  float t;
  glm::fvec2 uv;
  rayTriangleIntersection(ray, triangles[i], t, uv);

  ts[tid] = t;
  triIds[tid] = tid;
  __syncthreads();


  for (unsigned int s=1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      ts[tid] = fminf(ts[tid + s], ts[tid]);

      if (ts[tid] == ts[tid + s])
        triIds[tid] = triIds[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    *minT = ts[0];
    *tri = triIds[0];
  }
}

__device__ RaycastResult rayCast(const Ray& ray, const Node* bvh, const Triangle* triangles, const unsigned int nTriangles)
{
  float tMin = BIGT;
  int minTriIdx = -1;
  glm::fvec2 minUV;
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
      glm::fvec2 uv;

      for (int i = currentNode.startTri; i < currentNode.startTri + currentNode.nTri; ++i)
       {
        if (rayTriangleIntersection(ray, triangles[i], t, uv))
        {
          if(t < tMin)
          {
            tMin = t;
            minTriIdx = i;
            minUV = uv;
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

      if (leftHit && leftt < tMin)
      {
        stack[ptr] = currentNodeIdx + 1;
        ++ptr;
      }

      if (rightHit && rightt < tMin)
      {
        stack[ptr] = currentNode.rightIndex;
        ++ptr;
      }
    }

  }

  if (minTriIdx == -1)
    return result;

  glm::fvec3 hitPoint = ray.origin + glm::normalize(ray.direction) * tMin;

  result.point = hitPoint;
  result.t = tMin;
  result.triangleIdx = minTriIdx;
  result.uv = minUV;

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

  for (unsigned int i = 0; i < supersampling; ++i)
  {
    light.sample(pdf, lightSamplePoint, curandState1, curandState2);

    const glm::fvec3 interpolatedNormal = hitTriangle.normal(result.uv);
    glm::fvec3 shadowRayOrigin = result.point + interpolatedNormal * EPSILON;
    glm::fvec3 shadowRayDir = lightSamplePoint - shadowRayOrigin;

    float maxT = glm::length(shadowRayDir); // Distance to the light

    shadowRayDir = shadowRayDir / maxT;

    Ray shadowRay(shadowRayOrigin, glm::normalize(shadowRayDir));

    RaycastResult shadowResult = rayCast(shadowRay, bvh, triangles, nTriangles);

    if ((shadowResult && shadowResult.t >= maxT - EPSILON) || !shadowResult)
    {

      const float cosOmega = glm::clamp(glm::dot(shadowRayDir, interpolatedNormal), 0.f, 1.f);
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
    int secondaryLeft)
{
  glm::fvec3 color(0.f);

  // Primary ray
  RaycastResult result = rayCast(ray, bvh, triangles, nTriangles);

  if (!result)
    return color;

  const Triangle* hitTriangle = &triangles[result.triangleIdx];
  const Material* material = &materials[triangleMaterialIds[result.triangleIdx]];

  color = material->colorAmbient * 0.25f; // Ambient lightning
  color += material->colorDiffuse / glm::pi<float>() * areaLightShading(light, bvh, result, triangles, nTriangles, curandState1, curandState2, SHADOWSAMPLING);

  // For reflection rays
  Ray reflRay = ray;;
  RaycastResult reflResult = result;
  const Material* reflMaterial = &materials[triangleMaterialIds[result.triangleIdx]];
  const Triangle* reflHitTriangle = &triangles[result.triangleIdx];
  glm::fvec3 filterSpecular = material->colorSpecular;

  // For refraction rays
  Ray transOutRay = ray;
  RaycastResult transOutResult = result;
  const Material* transOutMaterial = &materials[triangleMaterialIds[result.triangleIdx]];
  const Triangle* transOutHitTriangle = &triangles[result.triangleIdx];
  glm::fvec3 filterTransparent = material->colorTransparent;


  // Secondary or reflections
  while (secondaryLeft)
  {
    const glm::fvec3 interpolatedNormalIn = transOutHitTriangle->normal(transOutResult.uv);

    // Transmittance and reflection according to fresnel
    const float cosi = fabsf(glm::dot(reflRay.direction, interpolatedNormalIn));
    const float sin2t = (AIR_INDEX / material->refrIdx) * (AIR_INDEX / material->refrIdx) * (1 - cosi * cosi);
    const float cost = sqrt(1 - sin2t);

    float Rp = (AIR_INDEX * cosi - material->refrIdx * cost) / (AIR_INDEX * cosi + material->refrIdx * cost);
    Rp = Rp * Rp;

    float Rs = (material->refrIdx * cosi - AIR_INDEX * cost) / (material->refrIdx * cosi + AIR_INDEX * cost);
    Rs = Rs * Rs;

    const float R = (Rp + Rs) * 0.5f;
    const float T = 1 - R;

    if (glm::length(filterSpecular) > 0.0f)
    {
      glm::fvec3 reflRayOrigin = reflResult.point + interpolatedNormalIn * EPSILON;
      glm::fvec3 reflRayDir = reflectionDirection(interpolatedNormalIn, reflRay.direction);
      reflRay = Ray(reflRayOrigin, reflRayDir);
      reflResult = rayCast(reflRay, bvh, triangles, nTriangles);

      if (!reflResult)
        break;

      reflHitTriangle = &triangles[result.triangleIdx];
      reflMaterial = &materials[triangleMaterialIds[reflResult.triangleIdx]];

      color += R * filterSpecular * reflMaterial->colorAmbient * 0.25f; // Ambient lightning
      color += R * filterSpecular * reflMaterial->colorDiffuse / glm::pi<float>() * areaLightShading(light, bvh, reflResult, triangles, nTriangles, curandState1, curandState2, SHADOWSAMPLING);

      filterSpecular = reflMaterial->colorSpecular;
    }

    if (glm::length(filterTransparent) > 0.0f && transOutResult.triangleIdx != -1)
    {
      const glm::fvec3 transInRayOrigin = transOutResult.point - interpolatedNormalIn * EPSILON;
      const glm::fvec3 transInRayDir = refractionDirection(interpolatedNormalIn, transOutRay.direction, AIR_INDEX, transOutMaterial->refrIdx);

      const Ray transInRay = Ray(transInRayOrigin, transInRayDir);
      const RaycastResult transInResult = rayCast(transInRay, bvh, triangles, nTriangles);

      if (!transInResult) // infinite volume?
        break;

      const Triangle& transInTriangle = triangles[transInResult.triangleIdx];
      const glm::fvec3 interpolatedNormalOut = transInTriangle.normal(transInResult.uv);
      const glm::fvec3 transOutRayOrigin = transInResult.point + interpolatedNormalOut * EPSILON;
      const glm::fvec3 transOutRayDir = refractionDirection(interpolatedNormalOut, transInRay.direction, material->refrIdx, AIR_INDEX);

      transOutRay = Ray(transOutRayOrigin, transOutRayDir);
      transOutResult = rayCast(transOutRay, bvh, triangles, nTriangles);

      if (!transOutResult)
        break;

      transOutHitTriangle = &triangles[transOutResult.triangleIdx];
      transOutMaterial = &materials[triangleMaterialIds[transOutResult.triangleIdx]];

      color += T * filterTransparent * transOutMaterial->colorAmbient * 0.25f; // Ambient lightning
      color += T * filterTransparent * transOutMaterial->colorDiffuse / glm::pi<float>() * areaLightShading(light, bvh, transOutResult, triangles, nTriangles, curandState1, curandState2, SHADOWSAMPLING);

      filterTransparent = transOutMaterial->colorTransparent;
    }

    --secondaryLeft;
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

__global__ void
__launch_bounds__(2 * BLOCKWIDTH * BLOCKWIDTH, 8)
cudaRender(\
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
      SECONDARY_RAYS);

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


