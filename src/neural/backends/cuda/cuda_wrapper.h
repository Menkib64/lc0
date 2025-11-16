/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#include <sstream>
#include <iomanip>
#include "utils/logging.h"

// Debug wrapper functions for CUDA, cuBLAS, and cuDNN calls
// These wrappers log function calls with their input and output parameters

namespace lczero {
namespace cudnn_backend {

// Enable/disable debug logging at compile time
#ifndef CUDA_WRAPPER_DEBUG
#define CUDA_WRAPPER_DEBUG 1
#endif

// Helper macro for conditional logging
#if CUDA_WRAPPER_DEBUG
#define CUDA_DEBUG_LOG(msg) LOGFILE << "[CUDA_WRAPPER] " << msg
#else
#define CUDA_DEBUG_LOG(msg) do {} while(0)
#endif

// Helper function to convert pointer to string
template <typename T>
inline std::string PtrToStr(T* ptr) {
  std::ostringstream oss;
  oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
  return oss.str();
}

// CUDA Runtime API Wrappers

template <typename T, typename S>
inline cudaError_t cudaMalloc(T** devPtr, S size) {
  CUDA_DEBUG_LOG("cudaMalloc(devPtr=" << PtrToStr(devPtr) 
                 << ", size=" << size << ")");
  cudaError_t result = ::cudaMalloc(reinterpret_cast<void**>(devPtr), size);
  CUDA_DEBUG_LOG("cudaMalloc -> " << cudaGetErrorString(result) 
                 << ", *devPtr=" << PtrToStr(*devPtr));
  return result;
}

template <typename T, typename S, typename F>
inline cudaError_t cudaHostAlloc(T** ptr, S size, F flags) {
  CUDA_DEBUG_LOG("cudaHostAlloc(ptr=" << PtrToStr(ptr) 
                 << ", size=" << size 
                 << ", flags=" << flags << ")");
  cudaError_t result = ::cudaHostAlloc(reinterpret_cast<void**>(ptr), size, flags);
  CUDA_DEBUG_LOG("cudaHostAlloc -> " << cudaGetErrorString(result) 
                 << ", *ptr=" << PtrToStr(*ptr));
  return result;
}

template <typename T>
inline cudaError_t cudaFree(T* devPtr) {
  CUDA_DEBUG_LOG("cudaFree(devPtr=" << PtrToStr(devPtr) << ")");
  cudaError_t result = ::cudaFree(devPtr);
  CUDA_DEBUG_LOG("cudaFree -> " << cudaGetErrorString(result));
  return result;
}

template <typename T>
inline cudaError_t cudaFreeHost(T* ptr) {
  CUDA_DEBUG_LOG("cudaFreeHost(ptr=" << PtrToStr(ptr) << ")");
  cudaError_t result = ::cudaFreeHost(ptr);
  CUDA_DEBUG_LOG("cudaFreeHost -> " << cudaGetErrorString(result));
  return result;
}

template <typename T1, typename T2>
inline cudaError_t cudaMemcpy(T1* dst, const T2* src, size_t count, 
                               cudaMemcpyKind kind) {
  [[maybe_unused]] const char* kind_str = 
      (kind == cudaMemcpyHostToDevice) ? "HostToDevice" :
      (kind == cudaMemcpyDeviceToHost) ? "DeviceToHost" :
      (kind == cudaMemcpyDeviceToDevice) ? "DeviceToDevice" :
      (kind == cudaMemcpyHostToHost) ? "HostToHost" : "Unknown";
  CUDA_DEBUG_LOG("cudaMemcpy(dst=" << PtrToStr(dst) 
                 << ", src=" << PtrToStr(src)
                 << ", count=" << count 
                 << ", kind=" << kind_str << ")");
  cudaError_t result = ::cudaMemcpy(dst, src, count, kind);
  CUDA_DEBUG_LOG("cudaMemcpy -> " << cudaGetErrorString(result));
  return result;
}

inline cudaError_t cudaMemcpyAsync_wrapper(void* dst, const void* src, size_t count,
                                   cudaMemcpyKind kind, cudaStream_t stream) {
  [[maybe_unused]] const char* kind_str = 
      (kind == cudaMemcpyHostToDevice) ? "HostToDevice" :
      (kind == cudaMemcpyDeviceToHost) ? "DeviceToHost" :
      (kind == cudaMemcpyDeviceToDevice) ? "DeviceToDevice" :
      (kind == cudaMemcpyHostToHost) ? "HostToHost" : "Unknown";
  CUDA_DEBUG_LOG("cudaMemcpyAsync(dst=" << PtrToStr(dst)
                 << ", src=" << PtrToStr(src)
                 << ", count=" << count
                 << ", kind=" << kind_str
                 << ", stream=" << PtrToStr(stream) << ")");
  cudaError_t result = ::cudaMemcpyAsync(dst, src, count, kind, stream);
  CUDA_DEBUG_LOG("cudaMemcpyAsync -> " << cudaGetErrorString(result));
  return result;
}
#define cudaMemcpyAsync cudaMemcpyAsync_wrapper

inline cudaError_t cudaMemset_wrapper(void* devPtr, int value, size_t count) {
  CUDA_DEBUG_LOG("cudaMemset(devPtr=" << PtrToStr(devPtr)
                 << ", value=" << value
                 << ", count=" << count << ")");
  cudaError_t result = ::cudaMemset(devPtr, value, count);
  CUDA_DEBUG_LOG("cudaMemset -> " << cudaGetErrorString(result));
  return result;
}
#define cudaMemset cudaMemset_wrapper

inline cudaError_t cudaMemsetAsync_wrapper(void* devPtr, int value, size_t count,
                                  cudaStream_t stream) {
  CUDA_DEBUG_LOG("cudaMemsetAsync(devPtr=" << PtrToStr(devPtr)
                 << ", value=" << value
                 << ", count=" << count
                 << ", stream=" << PtrToStr(stream) << ")");
  cudaError_t result = ::cudaMemsetAsync(devPtr, value, count, stream);
  CUDA_DEBUG_LOG("cudaMemsetAsync -> " << cudaGetErrorString(result));
  return result;
}

#define cudaMemsetAsync cudaMemsetAsync_wrapper

inline cudaError_t cudaGraphExecDestroy_wrapper(cudaGraphExec_t graphExec) {
  CUDA_DEBUG_LOG("cudaGraphExecDestroy(graphExec=" << PtrToStr(graphExec) << ")");
  cudaError_t result = ::cudaGraphExecDestroy(graphExec);
  CUDA_DEBUG_LOG("cudaGraphExecDestroy -> " << cudaGetErrorString(result));
  return result;
}

#define cudaGraphExecDestroy cudaGraphExecDestroy_wrapper

inline cudaError_t cudaGraphLaunch_wrapper(cudaGraphExec_t graphExec, 
                                            cudaStream_t stream) {
  CUDA_DEBUG_LOG("cudaGraphLaunch(graphExec=" << PtrToStr(graphExec)
                 << ", stream=" << PtrToStr(stream) << ")");
  cudaError_t result = ::cudaGraphLaunch(graphExec, stream);
  CUDA_DEBUG_LOG("cudaGraphLaunch -> " << cudaGetErrorString(result));
  return result;
}
#define cudaGraphLaunch cudaGraphLaunch_wrapper

inline cudaError_t cudaGraphInstantiate_wrapper(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                         cudaGraphNode_t* pErrorNode,
                                         char* pLogBuffer,
                                         size_t bufferSize) {
  CUDA_DEBUG_LOG("cudaGraphInstantiate(pGraphExec=" << PtrToStr(pGraphExec)
                 << ", graph=" << PtrToStr(graph)
                 << ", pErrorNode=" << PtrToStr(pErrorNode)
                 << ", pLogBuffer=" << PtrToStr(pLogBuffer)
                 << ", bufferSize=" << bufferSize << ")");
  cudaError_t result = ::cudaGraphInstantiate(pGraphExec, graph, pErrorNode,
                                             pLogBuffer, bufferSize);
  CUDA_DEBUG_LOG("cudaGraphInstantiate -> " << cudaGetErrorString(result)
                 << ", *pGraphExec=" << PtrToStr(*pGraphExec));
  return result;
}
#define cudaGraphInstantiate cudaGraphInstantiate_wrapper

template<typename Exec>
inline cudaError_t cudaGraphUpload_wrapper(Exec graphExec,
                                        cudaStream_t stream) {
  CUDA_DEBUG_LOG("cudaGraphUpload(graphExec=" << PtrToStr(graphExec)
                 << ", stream=" << PtrToStr(stream) << ")");
  cudaError_t result = ::cudaGraphUpload(graphExec, stream);
  CUDA_DEBUG_LOG("cudaGraphUpload -> " << cudaGetErrorString(result));
  return result;
}
#define cudaGraphUpload cudaGraphUpload_wrapper

inline cudaError_t cudaGraphDestroy_wrapper(cudaGraph_t graph) {
  CUDA_DEBUG_LOG("cudaGraphDestroy(graph=" << PtrToStr(graph) << ")");
  cudaError_t result = ::cudaGraphDestroy(graph);
  CUDA_DEBUG_LOG("cudaGraphDestroy -> " << cudaGetErrorString(result));
  return result;
}
#define cudaGraphDestroy cudaGraphDestroy_wrapper

inline cudaError_t cudaStreamCreate_wrapper(cudaStream_t* pStream) {
  CUDA_DEBUG_LOG("cudaStreamCreate(pStream=" << PtrToStr(pStream) << ")");
  cudaError_t result = ::cudaStreamCreate(pStream);
  CUDA_DEBUG_LOG("cudaStreamCreate -> " << cudaGetErrorString(result)
                 << ", *pStream=" << PtrToStr(*pStream));
  return result;
}
#define cudaStreamCreate cudaStreamCreate_wrapper

inline cudaError_t cudaStreamCreateWithFlags_wrapper(cudaStream_t* pStream, 
                                              unsigned int flags) {
  CUDA_DEBUG_LOG("cudaStreamCreateWithFlags(pStream=" << PtrToStr(pStream)
                 << ", flags=" << flags << ")");
  cudaError_t result = ::cudaStreamCreateWithFlags(pStream, flags);
  CUDA_DEBUG_LOG("cudaStreamCreateWithFlags -> " << cudaGetErrorString(result)
                 << ", *pStream=" << PtrToStr(*pStream));
  return result;
}
#define cudaStreamCreateWithFlags cudaStreamCreateWithFlags_wrapper

inline cudaError_t cudaStreamDestroy_wrapper(cudaStream_t stream) {
  CUDA_DEBUG_LOG("cudaStreamDestroy(stream=" << PtrToStr(stream) << ")");
  cudaError_t result = ::cudaStreamDestroy(stream);
  CUDA_DEBUG_LOG("cudaStreamDestroy -> " << cudaGetErrorString(result));
  return result;
}
#define cudaStreamDestroy cudaStreamDestroy_wrapper

inline cudaError_t cudaStreamSynchronize_wrapper(cudaStream_t stream) {
  CUDA_DEBUG_LOG("cudaStreamSynchronize(stream=" << PtrToStr(stream) << ")");
  cudaError_t result = ::cudaStreamSynchronize(stream);
  CUDA_DEBUG_LOG("cudaStreamSynchronize -> " << cudaGetErrorString(result));
  return result;
}
#define cudaStreamSynchronize cudaStreamSynchronize_wrapper

inline cudaError_t cudaStreamBeginCapture_wrapper(cudaStream_t stream, cudaStreamCaptureMode mode) {
  CUDA_DEBUG_LOG("cudaStreamBeginCapture(stream=" << PtrToStr(stream)
                 << ", mode=" << mode << ")");
  cudaError_t result = ::cudaStreamBeginCapture(stream, mode);
  CUDA_DEBUG_LOG("cudaStreamBeginCapture -> " << cudaGetErrorString(result));
  return result;
}
#define cudaStreamBeginCapture cudaStreamBeginCapture_wrapper

inline cudaError_t cudaStreamEndCapture_wrapper(cudaStream_t stream, cudaGraph_t* pGraph) {
  CUDA_DEBUG_LOG("cudaStreamEndCapture(stream=" << PtrToStr(stream)
                 << ", pGraph=" << PtrToStr(pGraph) << ")");
  cudaError_t result = ::cudaStreamEndCapture(stream, pGraph);
  CUDA_DEBUG_LOG("cudaStreamEndCapture -> " << cudaGetErrorString(result)
                 << ", *pGraph=" << PtrToStr(*pGraph));
  return result;
}
#define cudaStreamEndCapture cudaStreamEndCapture_wrapper

template <typename AttrId, typename AttrValue>
inline cudaError_t cudaStreamSetAttribute_wrapper(cudaStream_t stream,
                                           AttrId attr,
                                           const AttrValue* value) {
  CUDA_DEBUG_LOG("cudaStreamSetAttribute(stream=" << PtrToStr(stream)
                 << ", attr=" << attr << ")");
  cudaError_t result = ::cudaStreamSetAttribute(stream, attr, value);
  CUDA_DEBUG_LOG("cudaStreamSetAttribute -> " << cudaGetErrorString(result));
  return result;
}
#define cudaStreamSetAttribute cudaStreamSetAttribute_wrapper

inline cudaError_t cudaStreamWaitEvent_wrapper(cudaStream_t stream, cudaEvent_t event,
                                        unsigned int flags) {
  CUDA_DEBUG_LOG("cudaStreamWaitEvent(stream=" << PtrToStr(stream)
                 << ", event=" << PtrToStr(event)
                 << ", flags=" << flags << ")");
  cudaError_t result = ::cudaStreamWaitEvent(stream, event, flags);
  CUDA_DEBUG_LOG("cudaStreamWaitEvent -> " << cudaGetErrorString(result));
  return result;
}
#define cudaStreamWaitEvent cudaStreamWaitEvent_wrapper

inline cudaError_t cudaEventCreateWithFlags_wrapper(cudaEvent_t* event, 
                                             unsigned int flags) {
  CUDA_DEBUG_LOG("cudaEventCreateWithFlags(event=" << PtrToStr(event)
                 << ", flags=" << flags << ")");
  cudaError_t result = ::cudaEventCreateWithFlags(event, flags);
  CUDA_DEBUG_LOG("cudaEventCreateWithFlags -> " << cudaGetErrorString(result)
                 << ", *event=" << PtrToStr(*event));
  return result;
}
#define cudaEventCreateWithFlags cudaEventCreateWithFlags_wrapper

inline cudaError_t cudaEventDestroy_wrapper(cudaEvent_t event) {
  CUDA_DEBUG_LOG("cudaEventDestroy(event=" << PtrToStr(event) << ")");
  cudaError_t result = ::cudaEventDestroy(event);
  CUDA_DEBUG_LOG("cudaEventDestroy -> " << cudaGetErrorString(result));
  return result;
}
#define cudaEventDestroy cudaEventDestroy_wrapper

inline cudaError_t cudaEventRecord_wrapper(cudaEvent_t event, cudaStream_t stream) {
  CUDA_DEBUG_LOG("cudaEventRecord(event=" << PtrToStr(event)
                 << ", stream=" << PtrToStr(stream) << ")");
  cudaError_t result = ::cudaEventRecord(event, stream);
  CUDA_DEBUG_LOG("cudaEventRecord -> " << cudaGetErrorString(result));
  return result;
}
#define cudaEventRecord cudaEventRecord_wrapper

template <typename Event>
inline cudaError_t cudaEventRecordWithFlags_wrapper(Event event, 
                                             cudaStream_t stream,
                                             unsigned int flags) {
  CUDA_DEBUG_LOG("cudaEventRecordWithFlags(event=" << PtrToStr(event)
                 << ", stream=" << PtrToStr(stream)
                 << ", flags=" << flags << ")");
  cudaError_t result = ::cudaEventRecordWithFlags(event, stream, flags);
  CUDA_DEBUG_LOG("cudaEventRecordWithFlags -> " << cudaGetErrorString(result));
  return result;
}
#define cudaEventRecordWithFlags cudaEventRecordWithFlags_wrapper

inline cudaError_t cudaEventSynchronize_wrapper(cudaEvent_t event) {
  CUDA_DEBUG_LOG("cudaEventSynchronize(event=" << PtrToStr(event) << ")");
  cudaError_t result = ::cudaEventSynchronize(event);
  CUDA_DEBUG_LOG("cudaEventSynchronize -> " << cudaGetErrorString(result));
  return result;
}
#define cudaEventSynchronize cudaEventSynchronize_wrapper

inline cudaError_t cudaGetDeviceCount_wrapper(int* count) {
  CUDA_DEBUG_LOG("cudaGetDeviceCount(count=" << PtrToStr(count) << ")");
  cudaError_t result = ::cudaGetDeviceCount(count);
  CUDA_DEBUG_LOG("cudaGetDeviceCount -> " << cudaGetErrorString(result)
                 << ", *count=" << *count);
  return result;
}
#define cudaGetDeviceCount cudaGetDeviceCount_wrapper

inline cudaError_t cudaSetDevice_wrapper(int device) {
  CUDA_DEBUG_LOG("cudaSetDevice(device=" << device << ")");
  cudaError_t result = ::cudaSetDevice(device);
  CUDA_DEBUG_LOG("cudaSetDevice -> " << cudaGetErrorString(result));
  return result;
}
#define cudaSetDevice cudaSetDevice_wrapper

inline cudaError_t cudaGetDeviceProperties_wrapper(cudaDeviceProp* prop, int device) {
  CUDA_DEBUG_LOG("cudaGetDeviceProperties(prop=" << PtrToStr(prop)
                 << ", device=" << device << ")");
  cudaError_t result = ::cudaGetDeviceProperties(prop, device);
  CUDA_DEBUG_LOG("cudaGetDeviceProperties -> " << cudaGetErrorString(result)
                 << ", name=" << (result == cudaSuccess ? prop->name : "N/A"));
  return result;
}
#undef cudaGetDeviceProperties
#define cudaGetDeviceProperties cudaGetDeviceProperties_wrapper

inline cudaError_t cudaMemGetInfo_wrapper(size_t* free, size_t* total) {
  CUDA_DEBUG_LOG("cudaMemGetInfo(free=" << PtrToStr(free)
                 << ", total=" << PtrToStr(total) << ")");
  cudaError_t result = ::cudaMemGetInfo(free, total);
  CUDA_DEBUG_LOG("cudaMemGetInfo -> " << cudaGetErrorString(result)
                 << ", *free=" << *free
                 << ", *total=" << *total);
  return result;
}
#define cudaMemGetInfo cudaMemGetInfo_wrapper

inline cudaError_t cudaDeviceGetAttribute_wrapper(int* value, cudaDeviceAttr attr,
                                           int device) {
  CUDA_DEBUG_LOG("cudaDeviceGetAttribute(value=" << PtrToStr(value)
                 << ", attr=" << attr
                 << ", device=" << device << ")");
  cudaError_t result = ::cudaDeviceGetAttribute(value, attr, device);
  CUDA_DEBUG_LOG("cudaDeviceGetAttribute -> " << cudaGetErrorString(result)
                 << ", *value=" << *value);
  return result;
}
#define cudaDeviceGetAttribute cudaDeviceGetAttribute_wrapper

inline cudaError_t cudaRuntimeGetVersion_wrapper(int* runtimeVersion) {
  CUDA_DEBUG_LOG("cudaRuntimeGetVersion(runtimeVersion=" 
                 << PtrToStr(runtimeVersion) << ")");
  cudaError_t result = ::cudaRuntimeGetVersion(runtimeVersion);
  CUDA_DEBUG_LOG("cudaRuntimeGetVersion -> " << cudaGetErrorString(result)
                 << ", *runtimeVersion=" << *runtimeVersion);
  return result;
}
#define cudaRuntimeGetVersion cudaRuntimeGetVersion_wrapper

inline cudaError_t cudaDriverGetVersion_wrapper(int* driverVersion) {
  CUDA_DEBUG_LOG("cudaDriverGetVersion(driverVersion=" 
                 << PtrToStr(driverVersion) << ")");
  cudaError_t result = ::cudaDriverGetVersion(driverVersion);
  CUDA_DEBUG_LOG("cudaDriverGetVersion -> " << cudaGetErrorString(result)
                 << ", *driverVersion=" << *driverVersion);
  return result;
}
#define cudaDriverGetVersion cudaDriverGetVersion_wrapper

#if 0
inline cudaError_t cudaFuncSetAttribute_wrapper(const T* func, cudaFuncAttribute attr,
                                         int value) {
  CUDA_DEBUG_LOG("cudaFuncSetAttribute(func=" << PtrToStr(func)
                 << ", attr=" << attr
                 << ", value=" << value << ")");
  cudaError_t result = ::cudaFuncSetAttribute(func, attr, value);
  CUDA_DEBUG_LOG("cudaFuncSetAttribute -> " << cudaGetErrorString(result));
  return result;
}
#define cudaFuncSetAttribute cudaFuncSetAttribute_wrapper
#endif

template <typename T = void>
inline cudaError_t cudaCtxResetPersistingL2Cache_wrapper() {
  CUDA_DEBUG_LOG("cudaCtxResetPersistingL2Cache()");
  cudaError_t result = ::cudaCtxResetPersistingL2Cache();
  CUDA_DEBUG_LOG("cudaCtxResetPersistingL2Cache -> " << cudaGetErrorString(result));
  return result;
}
#define cudaCtxResetPersistingL2Cache cudaCtxResetPersistingL2Cache_wrapper

// cuBLAS API Wrappers

inline cublasStatus_t cublasCreate_wrapper(cublasHandle_t* handle) {
  CUDA_DEBUG_LOG("cublasCreate(handle=" << PtrToStr(handle) << ")");
  cublasStatus_t result = ::cublasCreate(handle);
  CUDA_DEBUG_LOG("cublasCreate -> " << static_cast<int>(result)
                 << ", *handle=" << PtrToStr(*handle));
  return result;
}
#undef cublasCreate
#define cublasCreate cublasCreate_wrapper

inline cublasStatus_t cublasDestroy_wrapper(cublasHandle_t handle) {
  CUDA_DEBUG_LOG("cublasDestroy(handle=" << PtrToStr(handle) << ")");
  cublasStatus_t result = ::cublasDestroy(handle);
  CUDA_DEBUG_LOG("cublasDestroy -> " << static_cast<int>(result));
  return result;
}
#undef cublasDestroy
#define cublasDestroy cublasDestroy_wrapper

inline cublasStatus_t cublasSetStream_wrapper(cublasHandle_t handle, 
                                       cudaStream_t streamId) {
  CUDA_DEBUG_LOG("cublasSetStream(handle=" << PtrToStr(handle)
                 << ", streamId=" << PtrToStr(streamId) << ")");
  cublasStatus_t result = ::cublasSetStream(handle, streamId);
  CUDA_DEBUG_LOG("cublasSetStream -> " << static_cast<int>(result));
  return result;
}
#undef cublasSetStream
#define cublasSetStream cublasSetStream_wrapper

inline cublasStatus_t cublasSetMathMode_wrapper(cublasHandle_t handle, 
                                         cublasMath_t mode) {
  CUDA_DEBUG_LOG("cublasSetMathMode(handle=" << PtrToStr(handle)
                 << ", mode=" << static_cast<int>(mode) << ")");
  cublasStatus_t result = ::cublasSetMathMode(handle, mode);
  CUDA_DEBUG_LOG("cublasSetMathMode -> " << static_cast<int>(result));
  return result;
}
#define cublasSetMathMode cublasSetMathMode_wrapper

inline cublasStatus_t cublasSgemm_wrapper(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k,
                                   const float* alpha, const float* A, int lda,
                                   const float* B, int ldb, const float* beta,
                                   float* C, int ldc) {
  CUDA_DEBUG_LOG("cublasSgemm(handle=" << PtrToStr(handle)
                 << ", transa=" << static_cast<int>(transa)
                 << ", transb=" << static_cast<int>(transb)
                 << ", m=" << m << ", n=" << n << ", k=" << k
                 << ", alpha=" << *alpha << ", lda=" << lda
                 << ", ldb=" << ldb << ", beta=" << *beta
                 << ", ldc=" << ldc << ")");
  cublasStatus_t result = ::cublasSgemm(handle, transa, transb, m, n, k,
                                        alpha, A, lda, B, ldb, beta, C, ldc);
  CUDA_DEBUG_LOG("cublasSgemm -> " << static_cast<int>(result));
  return result;
}
#undef cublasSgemm
#define cublasSgemm cublasSgemm_wrapper

inline cublasStatus_t cublasHgemm_wrapper(cublasHandle_t handle, cublasOperation_t transa,
                                   cublasOperation_t transb, int m, int n, int k,
                                   const __half* alpha, const __half* A, int lda,
                                   const __half* B, int ldb, const __half* beta,
                                   __half* C, int ldc) {
  CUDA_DEBUG_LOG("cublasHgemm(handle=" << PtrToStr(handle)
                 << ", transa=" << static_cast<int>(transa)
                 << ", transb=" << static_cast<int>(transb)
                 << ", m=" << m << ", n=" << n << ", k=" << k
                 << ", lda=" << lda << ", ldb=" << ldb
                 << ", ldc=" << ldc << ")");
  cublasStatus_t result = ::cublasHgemm(handle, transa, transb, m, n, k,
                                        alpha, A, lda, B, ldb, beta, C, ldc);
  CUDA_DEBUG_LOG("cublasHgemm -> " << static_cast<int>(result));
  return result;
}
#define cublasHgemm cublasHgemm_wrapper

inline cublasStatus_t cublasSgemmBatched_wrapper(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m, int n, int k,
                                          const float* alpha,
                                          const float* const Aarray[], int lda,
                                          const float* const Barray[], int ldb,
                                          const float* beta,
                                          float* const Carray[], int ldc,
                                          int batchCount) {
  CUDA_DEBUG_LOG("cublasSgemmBatched(handle=" << PtrToStr(handle)
                 << ", m=" << m << ", n=" << n << ", k=" << k
                 << ", batchCount=" << batchCount << ")");
  cublasStatus_t result = ::cublasSgemmBatched(handle, transa, transb, m, n, k,
                                               alpha, Aarray, lda, Barray, ldb,
                                               beta, Carray, ldc, batchCount);
  CUDA_DEBUG_LOG("cublasSgemmBatched -> " << static_cast<int>(result));
  return result;
}
#define cublasSgemmBatched cublasSgemmBatched_wrapper

inline cublasStatus_t cublasHgemmBatched_wrapper(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m, int n, int k,
                                          const __half* alpha,
                                          const __half* const Aarray[], int lda,
                                          const __half* const Barray[], int ldb,
                                          const __half* beta,
                                          __half* const Carray[], int ldc,
                                          int batchCount) {
  CUDA_DEBUG_LOG("cublasHgemmBatched(handle=" << PtrToStr(handle)
                 << ", m=" << m << ", n=" << n << ", k=" << k
                 << ", batchCount=" << batchCount << ")");
  cublasStatus_t result = ::cublasHgemmBatched(handle, transa, transb, m, n, k,
                                               alpha, Aarray, lda, Barray, ldb,
                                               beta, Carray, ldc, batchCount);
  CUDA_DEBUG_LOG("cublasHgemmBatched -> " << static_cast<int>(result));
  return result;
}
#define cublasHgemmBatched cublasHgemmBatched_wrapper

inline cublasStatus_t cublasSgemmStridedBatched_wrapper(cublasHandle_t handle,
                                                 cublasOperation_t transa,
                                                 cublasOperation_t transb,
                                                 int m, int n, int k,
                                                 const float* alpha,
                                                 const float* A, int lda,
                                                 long long int strideA,
                                                 const float* B, int ldb,
                                                 long long int strideB,
                                                 const float* beta,
                                                 float* C, int ldc,
                                                 long long int strideC,
                                                 int batchCount) {
  CUDA_DEBUG_LOG("cublasSgemmStridedBatched(handle=" << PtrToStr(handle)
                 << ", m=" << m << ", n=" << n << ", k=" << k
                 << ", batchCount=" << batchCount << ")");
  cublasStatus_t result = ::cublasSgemmStridedBatched(handle, transa, transb,
                                                      m, n, k, alpha, A, lda,
                                                      strideA, B, ldb, strideB,
                                                      beta, C, ldc, strideC,
                                                      batchCount);
  CUDA_DEBUG_LOG("cublasSgemmStridedBatched -> " << static_cast<int>(result));
  return result;
}
#define cublasSgemmStridedBatched cublasSgemmStridedBatched_wrapper

template <typename T1, typename T2, typename T3, typename T4, typename ComputeType>
inline cublasStatus_t cublasGemmStridedBatchedEx_wrapper(cublasHandle_t handle,
                                                  cublasOperation_t transa,
                                                  cublasOperation_t transb,
                                                  int m, int n, int k,
                                                  const T1* alpha,
                                                  const T2* A, cudaDataType Atype,
                                                  int lda, long long int strideA,
                                                  const T3* B, cudaDataType Btype,
                                                  int ldb, long long int strideB,
                                                  const T1* beta,
                                                  T4* C, cudaDataType Ctype,
                                                  int ldc, long long int strideC,
                                                  int batchCount,
                                                  ComputeType computeType,
                                                  cublasGemmAlgo_t algo) {
  CUDA_DEBUG_LOG("cublasGemmStridedBatchedEx(handle=" << PtrToStr(handle)
                 << ", m=" << m << ", n=" << n << ", k=" << k
                 << ", batchCount=" << batchCount << ")");
  cublasStatus_t result = ::cublasGemmStridedBatchedEx(handle, transa, transb,
                                                       m, n, k, alpha, A, Atype,
                                                       lda, strideA, B, Btype,
                                                       ldb, strideB, beta, C,
                                                       Ctype, ldc, strideC,
                                                       batchCount, computeType,
                                                       algo);
  CUDA_DEBUG_LOG("cublasGemmStridedBatchedEx -> " << static_cast<int>(result));
  return result;
}
#define cublasGemmStridedBatchedEx cublasGemmStridedBatchedEx_wrapper

// cuDNN API Wrappers

#ifdef USE_CUDNN

inline cudnnStatus_t cudnnCreate_wrapper(cudnnHandle_t* handle) {
  CUDA_DEBUG_LOG("cudnnCreate(handle=" << PtrToStr(handle) << ")");
  cudnnStatus_t result = ::cudnnCreate(handle);
  CUDA_DEBUG_LOG("cudnnCreate -> " << cudnnGetErrorString(result)
                 << ", *handle=" << PtrToStr(*handle));
  return result;
}
#define cudnnCreate cudnnCreate_wrapper

inline cudnnStatus_t cudnnDestroy_wrapper(cudnnHandle_t handle) {
  CUDA_DEBUG_LOG("cudnnDestroy(handle=" << PtrToStr(handle) << ")");
  cudnnStatus_t result = ::cudnnDestroy(handle);
  CUDA_DEBUG_LOG("cudnnDestroy -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnDestroy cudnnDestroy_wrapper

inline cudnnStatus_t cudnnSetStream_wrapper(cudnnHandle_t handle, cudaStream_t streamId) {
  CUDA_DEBUG_LOG("cudnnSetStream(handle=" << PtrToStr(handle)
                 << ", streamId=" << PtrToStr(streamId) << ")");
  cudnnStatus_t result = ::cudnnSetStream(handle, streamId);
  CUDA_DEBUG_LOG("cudnnSetStream -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnSetStream cudnnSetStream_wrapper

inline size_t cudnnGetVersion_wrapper() {
  size_t version = ::cudnnGetVersion();
  CUDA_DEBUG_LOG("cudnnGetVersion() -> " << version);
  return version;
}
#define cudnnGetVersion cudnnGetVersion_wrapper

inline cudnnStatus_t cudnnCreateTensorDescriptor_wrapper(cudnnTensorDescriptor_t* tensorDesc) {
  CUDA_DEBUG_LOG("cudnnCreateTensorDescriptor(tensorDesc=" 
                 << PtrToStr(tensorDesc) << ")");
  cudnnStatus_t result = ::cudnnCreateTensorDescriptor(tensorDesc);
  CUDA_DEBUG_LOG("cudnnCreateTensorDescriptor -> " << cudnnGetErrorString(result)
                 << ", *tensorDesc=" << PtrToStr(*tensorDesc));
  return result;
}
#define cudnnCreateTensorDescriptor cudnnCreateTensorDescriptor_wrapper

inline cudnnStatus_t cudnnDestroyTensorDescriptor_wrapper(cudnnTensorDescriptor_t tensorDesc) {
  CUDA_DEBUG_LOG("cudnnDestroyTensorDescriptor(tensorDesc=" 
                 << PtrToStr(tensorDesc) << ")");
  cudnnStatus_t result = ::cudnnDestroyTensorDescriptor(tensorDesc);
  CUDA_DEBUG_LOG("cudnnDestroyTensorDescriptor -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnDestroyTensorDescriptor cudnnDestroyTensorDescriptor_wrapper

inline cudnnStatus_t cudnnSetTensor4dDescriptor_wrapper(cudnnTensorDescriptor_t tensorDesc,
                                                  cudnnTensorFormat_t format,
                                                  cudnnDataType_t dataType,
                                                  int n, int c, int h, int w) {
  CUDA_DEBUG_LOG("cudnnSetTensor4dDescriptor(tensorDesc=" << PtrToStr(tensorDesc)
                 << ", format=" << static_cast<int>(format)
                 << ", dataType=" << static_cast<int>(dataType)
                 << ", n=" << n << ", c=" << c
                 << ", h=" << h << ", w=" << w << ")");
  cudnnStatus_t result = ::cudnnSetTensor4dDescriptor(tensorDesc, format, dataType,
                                                       n, c, h, w);
  CUDA_DEBUG_LOG("cudnnSetTensor4dDescriptor -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnSetTensor4dDescriptor cudnnSetTensor4dDescriptor_wrapper

inline cudnnStatus_t cudnnCreateFilterDescriptor_wrapper(cudnnFilterDescriptor_t* filterDesc) {
  CUDA_DEBUG_LOG("cudnnCreateFilterDescriptor(filterDesc=" 
                 << PtrToStr(filterDesc) << ")");
  cudnnStatus_t result = ::cudnnCreateFilterDescriptor(filterDesc);
  CUDA_DEBUG_LOG("cudnnCreateFilterDescriptor -> " << cudnnGetErrorString(result)
                 << ", *filterDesc=" << PtrToStr(*filterDesc));
  return result;
}
#define cudnnCreateFilterDescriptor cudnnCreateFilterDescriptor_wrapper

inline cudnnStatus_t cudnnDestroyFilterDescriptor_wrapper(cudnnFilterDescriptor_t filterDesc) {
  CUDA_DEBUG_LOG("cudnnDestroyFilterDescriptor(filterDesc=" 
                 << PtrToStr(filterDesc) << ")");
  cudnnStatus_t result = ::cudnnDestroyFilterDescriptor(filterDesc);
  CUDA_DEBUG_LOG("cudnnDestroyFilterDescriptor -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnDestroyFilterDescriptor cudnnDestroyFilterDescriptor_wrapper

inline cudnnStatus_t cudnnSetFilter4dDescriptor_wrapper(cudnnFilterDescriptor_t filterDesc,
                                                  cudnnDataType_t dataType,
                                                  cudnnTensorFormat_t format,
                                                  int k, int c, int h, int w) {
  CUDA_DEBUG_LOG("cudnnSetFilter4dDescriptor(filterDesc=" << PtrToStr(filterDesc)
                 << ", dataType=" << static_cast<int>(dataType)
                 << ", format=" << static_cast<int>(format)
                 << ", k=" << k << ", c=" << c
                 << ", h=" << h << ", w=" << w << ")");
  cudnnStatus_t result = ::cudnnSetFilter4dDescriptor(filterDesc, dataType, format,
                                                       k, c, h, w);
  CUDA_DEBUG_LOG("cudnnSetFilter4dDescriptor -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnSetFilter4dDescriptor cudnnSetFilter4dDescriptor_wrapper

inline cudnnStatus_t cudnnCreateConvolutionDescriptor_wrapper(
    cudnnConvolutionDescriptor_t* convDesc) {
  CUDA_DEBUG_LOG("cudnnCreateConvolutionDescriptor(convDesc=" 
                 << PtrToStr(convDesc) << ")");
  cudnnStatus_t result = ::cudnnCreateConvolutionDescriptor(convDesc);
  CUDA_DEBUG_LOG("cudnnCreateConvolutionDescriptor -> " << cudnnGetErrorString(result)
                 << ", *convDesc=" << PtrToStr(*convDesc));
  return result;
}
#define cudnnCreateConvolutionDescriptor cudnnCreateConvolutionDescriptor_wrapper

template <typename Convolution> inline cudnnStatus_t cudnnDestroyConvolutionDescriptor_wrapper(
    Convolution convDesc) {
  CUDA_DEBUG_LOG("cudnnDestroyConvolutionDescriptor(convDesc=" 
                 << PtrToStr(convDesc) << ")");
  cudnnStatus_t result = ::cudnnDestroyConvolutionDescriptor(convDesc);
  CUDA_DEBUG_LOG("cudnnDestroyConvolutionDescriptor -> " 
                 << cudnnGetErrorString(result));
  return result;
}
#define cudnnDestroyConvolutionDescriptor cudnnDestroyConvolutionDescriptor_wrapper

template <typename Convolution> inline cudnnStatus_t cudnnSetConvolution2dDescriptor_wrapper(
    Convolution convDesc,
    int pad_h, int pad_w, int u, int v, int dilation_h, int dilation_w,
    cudnnConvolutionMode_t mode, cudnnDataType_t computeType) {
  CUDA_DEBUG_LOG("cudnnSetConvolution2dDescriptor(convDesc=" << PtrToStr(convDesc)
                 << ", pad_h=" << pad_h << ", pad_w=" << pad_w
                 << ", u=" << u << ", v=" << v
                 << ", dilation_h=" << dilation_h << ", dilation_w=" << dilation_w
                 << ", mode=" << static_cast<int>(mode)
                 << ", computeType=" << static_cast<int>(computeType) << ")");
  cudnnStatus_t result = ::cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w,
                                                           u, v, dilation_h,
                                                           dilation_w, mode,
                                                           computeType);
  CUDA_DEBUG_LOG("cudnnSetConvolution2dDescriptor -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnSetConvolution2dDescriptor cudnnSetConvolution2dDescriptor_wrapper

template <typename Convolution> inline cudnnStatus_t cudnnSetConvolutionMathType_wrapper(
    Convolution convDesc, cudnnMathType_t mathType) {
  CUDA_DEBUG_LOG("cudnnSetConvolutionMathType(convDesc=" << PtrToStr(convDesc)
                 << ", mathType=" << static_cast<int>(mathType) << ")");
  cudnnStatus_t result = ::cudnnSetConvolutionMathType(convDesc, mathType);
  CUDA_DEBUG_LOG("cudnnSetConvolutionMathType -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnSetConvolutionMathType cudnnSetConvolutionMathType_wrapper

template <typename Activation> inline cudnnStatus_t cudnnCreateActivationDescriptor_wrapper(
    Activation* activationDesc) {
  CUDA_DEBUG_LOG("cudnnCreateActivationDescriptor(activationDesc=" 
                 << PtrToStr(activationDesc) << ")");
  cudnnStatus_t result = ::cudnnCreateActivationDescriptor(activationDesc);
  CUDA_DEBUG_LOG("cudnnCreateActivationDescriptor -> " << cudnnGetErrorString(result)
                 << ", *activationDesc=" << PtrToStr(*activationDesc));
  return result;
}
#define cudnnCreateActivationDescriptor cudnnCreateActivationDescriptor_wrapper

template <typename Activation> inline cudnnStatus_t cudnnDestroyActivationDescriptor_wrapper(
    Activation activationDesc) {
  CUDA_DEBUG_LOG("cudnnDestroyActivationDescriptor(activationDesc=" 
                 << PtrToStr(activationDesc) << ")");
  cudnnStatus_t result = ::cudnnDestroyActivationDescriptor(activationDesc);
  CUDA_DEBUG_LOG("cudnnDestroyActivationDescriptor -> " 
                 << cudnnGetErrorString(result));
  return result;
}
#define cudnnDestroyActivationDescriptor cudnnDestroyActivationDescriptor_wrapper

template <typename Activation> inline cudnnStatus_t cudnnSetActivationDescriptor_wrapper(
    Activation activationDesc,
    cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef) {
  CUDA_DEBUG_LOG("cudnnSetActivationDescriptor(activationDesc=" 
                 << PtrToStr(activationDesc)
                 << ", mode=" << static_cast<int>(mode)
                 << ", reluNanOpt=" << static_cast<int>(reluNanOpt)
                 << ", coef=" << coef << ")");
  cudnnStatus_t result = ::cudnnSetActivationDescriptor(activationDesc, mode,
                                                        reluNanOpt, coef);
  CUDA_DEBUG_LOG("cudnnSetActivationDescriptor -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnSetActivationDescriptor cudnnSetActivationDescriptor_wrapper

template <typename Handle> inline cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize_wrapper(
    Handle handle,
    const cudnnTensorDescriptor_t xDesc,
    const cudnnFilterDescriptor_t wDesc,
    const cudnnConvolutionDescriptor_t convDesc,
    const cudnnTensorDescriptor_t yDesc,
    cudnnConvolutionFwdAlgo_t algo,
    size_t* sizeInBytes) {
  CUDA_DEBUG_LOG("cudnnGetConvolutionForwardWorkspaceSize(handle=" 
                 << PtrToStr(handle) << ", algo=" << static_cast<int>(algo) << ")");
  cudnnStatus_t result = ::cudnnGetConvolutionForwardWorkspaceSize(
      handle, xDesc, wDesc, convDesc, yDesc, algo, sizeInBytes);
  CUDA_DEBUG_LOG("cudnnGetConvolutionForwardWorkspaceSize -> " 
                 << cudnnGetErrorString(result)
                 << ", *sizeInBytes=" << *sizeInBytes);
  return result;
}
#define cudnnGetConvolutionForwardWorkspaceSize cudnnGetConvolutionForwardWorkspaceSize_wrapper

template <typename T1, typename T2, typename T3, typename T4, typename T5>
inline cudnnStatus_t cudnnConvolutionForward_wrapper(
    cudnnHandle_t handle,
    const T1* alpha,
    const cudnnTensorDescriptor_t xDesc, const T2* x,
    const cudnnFilterDescriptor_t wDesc, const T3* w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    T4* workSpace, size_t workSpaceSizeInBytes,
    const T1* beta,
    const cudnnTensorDescriptor_t yDesc, T5* y) {
  CUDA_DEBUG_LOG("cudnnConvolutionForward(handle=" << PtrToStr(handle)
                 << ", algo=" << static_cast<int>(algo)
                 << ", workSpaceSizeInBytes=" << workSpaceSizeInBytes << ")");
  cudnnStatus_t result = ::cudnnConvolutionForward(handle, alpha, xDesc, x,
                                                   wDesc, w, convDesc, algo,
                                                   workSpace, workSpaceSizeInBytes,
                                                   beta, yDesc, y);
  CUDA_DEBUG_LOG("cudnnConvolutionForward -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnConvolutionForward cudnnConvolutionForward_wrapper

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
inline cudnnStatus_t cudnnConvolutionBiasActivationForward_wrapper(
    cudnnHandle_t handle,
    const T1* alpha1,
    const cudnnTensorDescriptor_t xDesc, const T2* x,
    const cudnnFilterDescriptor_t wDesc, const T3* w,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    T4* workSpace, size_t workSpaceSizeInBytes,
    const T1* alpha2,
    const cudnnTensorDescriptor_t zDesc, const T5* z,
    const cudnnTensorDescriptor_t biasDesc, const T6* bias,
    const cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t yDesc, T7* y) {
  CUDA_DEBUG_LOG("cudnnConvolutionBiasActivationForward(handle=" << PtrToStr(handle)
                 << ", algo=" << static_cast<int>(algo)
                 << ", workSpaceSizeInBytes=" << workSpaceSizeInBytes << ")");
  cudnnStatus_t result = ::cudnnConvolutionBiasActivationForward(
      handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, alpha2, zDesc, z, biasDesc, bias, activationDesc,
      yDesc, y);
  CUDA_DEBUG_LOG("cudnnConvolutionBiasActivationForward -> " 
                 << cudnnGetErrorString(result));
  return result;
}
#define cudnnConvolutionBiasActivationForward cudnnConvolutionBiasActivationForward_wrapper

template <typename T1, typename T2, typename T3>
inline cudnnStatus_t cudnnActivationForward_wrapper(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t activationDesc,
    const T1* alpha,
    const cudnnTensorDescriptor_t xDesc, const T2* x,
    const T1* beta,
    const cudnnTensorDescriptor_t yDesc, T3* y) {
  CUDA_DEBUG_LOG("cudnnActivationForward(handle=" << PtrToStr(handle) << ")");
  cudnnStatus_t result = ::cudnnActivationForward(handle, activationDesc, alpha,
                                                  xDesc, x, beta, yDesc, y);
  CUDA_DEBUG_LOG("cudnnActivationForward -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnActivationForward cudnnActivationForward_wrapper

template <typename T1, typename T2, typename T3>
inline cudnnStatus_t cudnnAddTensor_wrapper(
    cudnnHandle_t handle,
    const T1* alpha,
    const cudnnTensorDescriptor_t aDesc, const T2* A,
    const T1* beta,
    const cudnnTensorDescriptor_t cDesc, T3* C) {
  CUDA_DEBUG_LOG("cudnnAddTensor(handle=" << PtrToStr(handle) << ")");
  cudnnStatus_t result = ::cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  CUDA_DEBUG_LOG("cudnnAddTensor -> " << cudnnGetErrorString(result));
  return result;
}
#define cudnnAddTensor cudnnAddTensor_wrapper

#endif  // USE_CUDNN

// Kernel Launch Logging Macros
// These macros wrap CUDA kernel launches to log execution parameters

#if CUDA_WRAPPER_DEBUG
// Log kernel launch with grid, block, shared memory, and stream information
#define CUDA_KERNEL_LAUNCH_LOG(kernel_name, grid_dim, block_dim, shared_mem, stream_ptr) \
  do { \
    dim3 _grid = (grid_dim); \
    dim3 _block = (block_dim); \
    LOGFILE << "[CUDA_WRAPPER] Kernel launch: " << #kernel_name \
            << " grid=(" << _grid.x << "," << _grid.y << "," << _grid.z << ")" \
            << " block=(" << _block.x << "," << _block.y << "," << _block.z << ")" \
            << " smem=" << (shared_mem) \
            << " stream=" << PtrToStr(stream_ptr); \
  } while(0)
#else
#define CUDA_KERNEL_LAUNCH_LOG(kernel_name, grid_dim, block_dim, shared_mem, stream_ptr) do {} while(0)
#endif

}  // namespace cudnn_backend
}  // namespace lczero
