/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include "lc0ex_cuda.h"

#include <absl/container/inlined_vector.h>
#include <cuda.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "neural/network.h"
#include "proto/lc0ex.pb.h"
#include "proto/lc0ex_metadata.pb.h"
#include "utils/exception.h"
#include "utils/trace.h"

// TODO: Move these to a separate shared implementation file.
namespace lczero::lc0ex {
void MemoryBase::IsValidRange(std::string_view name, ptrdiff_t offset,
                              size_t size
#if __cpp_lib_source_location >= 201907L
                              ,
                              const std::source_location& location
#endif
) const {
  if (offset < 0 || size == 0 || offset + size > size_) {
    std::ostringstream oss;
    oss << "Invalid range for buffer " << name << ": offset=" << offset
        << ", size=" << size << ", buffer_size=" << size_
#if __cpp_lib_source_location >= 201907L
        << ", location=" << location.file_name() << ":" << location.line()
        << " in " << location.function_name()
#endif
        ;
    throw Exception(oss.str());
  }
}

NodeBase::NodeBase(const pblczero::Node& node)
    : dependencies_(node.dependencies().begin(), node.dependencies().end()),
      priority_(node.priority()) {}
}  // namespace lczero::lc0ex

namespace lczero::lc0ex::cuda {

namespace {
constexpr std::uint32_t kMagic = 0x1c0e;
constexpr std::uint32_t kFormat = 1;

[[noreturn]] void ThrowCuda(CUresult status, const char* expression,
                            const char* file, int line) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &description);

  std::string message = expression;
  message += " failed at ";
  message += file;
  message += ":";
  message += std::to_string(line);
  message += ": ";
  message += name ? name : "unknown CUDA error";
  if (description) {
    message += " (";
    message += description;
    message += ")";
  }
  throw Exception(message);
}

#define LC0EX_CUDA_CHECK(expression)                            \
  do {                                                          \
    const CUresult lc0ex_status = (expression);                 \
    if (lc0ex_status != CUDA_SUCCESS)                           \
      ThrowCuda(lc0ex_status, #expression, __FILE__, __LINE__); \
  } while (false)

std::size_t ElementSize(pblczero::Buffer::DataType type) {
  switch (type) {
    case pblczero::Buffer::DATA_TYPE_F32:
      return sizeof(float);
    case pblczero::Buffer::DATA_TYPE_U8:
      return sizeof(std::uint8_t);
    case pblczero::Buffer::DATA_TYPE_F16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_U64:
      return sizeof(std::uint64_t);
    case pblczero::Buffer::DATA_TYPE_BF16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_UNKNOWN:
      break;
  }
  throw Exception("Unsupported or unknown buffer data type.");
}

// Helper class to add extra implementation functions which aren't exposed in
// the header file.
class ExecutableImpl : public CudaExecutable {
 public:
  void BuildModules(const pblczero::NeuralExecutable& source) {
    LCTRACE_FUNCTION_SCOPE;
    this->modules_.reserve(source.binaries_size());
    for (const auto& binary : source.binaries()) {
      CUmodule module = nullptr;
      LC0EX_CUDA_CHECK(cuModuleLoadData(&module, binary.data().data()));
      this->modules_.push_back(reinterpret_cast<Module>(module));
    }
  }

  void BuildPersistentAllocation(const pblczero::NeuralExecutable& source) {
    if (source.has_persistent_allocation()) {
      if (!source.persistent_allocation().has_size_bytes()) {
        throw Exception("Persistent allocation must have size_bytes.");
      }
      if (!source.persistent_allocation().has_alignment_bytes()) {
        throw Exception("Persistent allocation must have alignment_bytes.");
      }
      size_t size = source.persistent_allocation().size_bytes();
      size_t alignment = source.persistent_allocation().alignment_bytes();
      this->persistent_allocation_ = CudaMemory(size, alignment);
    }
  }

  template <typename T>
  void BuildExecutionAllocation(const T& source) {
    if (source.has_execution_allocation()) {
      if (!source.execution_allocation().has_size_bytes()) {
        throw Exception("Execution allocation must have size_bytes.");
      }
      if (!source.execution_allocation().has_alignment_bytes()) {
        throw Exception("Execution allocation must have alignment_bytes.");
      }
      if (source.execution_allocation().size_bytes() <
          execution_allocation_.size_bytes_) {
        return;
      }
      execution_allocation_.size_bytes_ =
          source.execution_allocation().size_bytes();
      execution_allocation_.alignment_bytes_ =
          source.execution_allocation().alignment_bytes();
    }
  }

  void BuildPrograms(const pblczero::NeuralExecutable& source) {
    this->programs_.reserve(source.programs_size());
    for (const auto& program : source.programs()) {
      const auto name = std::string(program.name());

      this->programs_.emplace_back(program, *this, source.kernels());
    }
  }

  CUmodule GetBinary(size_t idx) const {
    if (idx >= this->modules_.size()) {
      throw Exception("Binary index out of range.");
    }
    return reinterpret_cast<CUmodule>(this->modules_[idx]);
  }
};

}  // namespace

// The helper class to maintain state for graph node execution to a CUDA
// stream.
template <typename T>
class LaunchState {
 public:
  using ComputeType = T;
  static constexpr bool is_cuda_capturing = false;
  ComputationState<CudaRuntime, T>& cs_;
  size_t batch_size;
  const CudaMemory& persistent_memory_;
  CudaEvent& compute_ordering_event_;

  absl::InlinedVector<uint64_t, 8> argval;
  absl::InlinedVector<void*, 8> argptr;
};

// The helper class to maintain state for graph capture using CUDA graph API. It
// can build either a linear or a DAG graph depending on the graph mode.
template <typename T>
class CaptureState : public LaunchState<T> {
 public:
  static constexpr bool is_cuda_capturing = true;
  GraphCapture graph_;
  std::vector<CUgraphNode> graph_nodes_;
  absl::InlinedVector<CUgraphNode, 4> dependencies_;
};

// Kernel argument implementation. Each type implements its own kernel argument
// logic based on the Triton node arguments.
template <typename State>
void ArgumentNull::operator()(void*& arg, std::uint64_t& value,
                              const State&) const {
  arg = static_cast<void*>(&value);
  value = 0;
}

template <typename State>
void ArgumentSymbol::operator()(void*& arg, std::uint64_t& value,
                                const State&) const {
  arg = static_cast<void*>(&value);
  value = reinterpret_cast<std::uint64_t>(symbol_);
}

template <typename State>
void ArgumentPersistentBuffer::operator()(void*& arg, std::uint64_t& value,
                                          const State& state) const {
  arg = static_cast<void*>(&value);
  value = reinterpret_cast<std::uint64_t>(state.persistent_memory_.Data() +
                                          offset_);
}

template <typename State>
void ArgumentExecutionBuffer::operator()(void*& arg, std::uint64_t& value,
                                         const State& state) const {
  arg = static_cast<void*>(&value);
  value = reinterpret_cast<std::uint64_t>(state.cs_.device_memory_.Data() +
                                          offset_);
}

// The CudaStream class implementation. It wraps a CUDA stream and provides
// methods to manage its lifecycle and operations.
CudaStream::CudaStream(StreamFlags flags) {
  int cuda_flags = CU_STREAM_NON_BLOCKING;
  if (flags != StreamFlags::DEFAULT) {
    throw Exception("Unsupported stream flags (" +
                    std::to_string(static_cast<int>(flags)) + ").");
  }
  CUstream stream = nullptr;
  LC0EX_CUDA_CHECK(cuStreamCreate(&stream, cuda_flags));
  stream_ = reinterpret_cast<Stream>(stream);
}

CudaStream::CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}
CudaStream& CudaStream::operator=(CudaStream&& other) noexcept {
  if (this != &other) {
    if (stream_ != nullptr) {
      LC0EX_CUDA_CHECK(cuStreamDestroy(*this));
    }
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) {
    LC0EX_CUDA_CHECK(cuStreamDestroy(*this));
  }
}

template <typename StreamType>
CudaStream::operator StreamType() const {
  static_assert(std::is_same_v<StreamType, CUstream>,
                "StreamType must be CUstream");
  return reinterpret_cast<CUstream>(stream_);
}

bool CudaStream::IsIdle() const {
  assert(stream_ != nullptr);
  CUresult status = cuStreamQuery(*this);
  if (status == CUDA_SUCCESS) return true;
  if (status == CUDA_ERROR_NOT_READY) return false;
  ThrowCuda(status, "cuStreamQuery", __FILE__, __LINE__);
}

void CudaStream::Synchronize() const {
  assert(stream_ != nullptr);
  LC0EX_CUDA_CHECK(cuStreamSynchronize(*this));
}

void CudaStream::WaitEvent(CudaEvent& event) const {
  assert(stream_ != nullptr);
  assert(event);
  LC0EX_CUDA_CHECK(cuStreamWaitEvent(*this, event, 0));
}

void CudaStream::RecordEvent(CudaEvent& event) const {
  assert(stream_ != nullptr);
  assert(event);
  LC0EX_CUDA_CHECK(cuEventRecord(event, *this));
}

// The CudaEvent class implementation. It wraps a CUDA event and provides
// methods to manage its lifecycle and operations.
CudaEvent::CudaEvent(EventFlags flags) {
  int cuda_flags = 0;
  if ((flags & EventFlags::BLOCKING) == EventFlags::BLOCKING) {
    cuda_flags |= CU_EVENT_BLOCKING_SYNC;
    flags = flags & ~EventFlags::BLOCKING;
  }
  if ((flags & EventFlags::USE_TIMING) == EventFlags::USE_TIMING) {
    flags = flags & ~EventFlags::USE_TIMING;
  } else {
    cuda_flags |= CU_EVENT_DISABLE_TIMING;
  }
  if (flags != EventFlags::DEFAULT) {
    throw Exception("Unsupported event flags (" +
                    std::to_string(static_cast<int>(flags)) + ").");
  }
  CUevent event = nullptr;
  LC0EX_CUDA_CHECK(cuEventCreate(&event, cuda_flags));
  event_ = reinterpret_cast<Event>(event);
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
  other.event_ = nullptr;
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
  if (this != &other) {
    if (event_ != nullptr) {
      LC0EX_CUDA_CHECK(cuEventDestroy(*this));
    }
    event_ = other.event_;
    other.event_ = nullptr;
  }
  return *this;
}

CudaEvent::~CudaEvent() {
  if (event_ != nullptr) {
    LC0EX_CUDA_CHECK(cuEventDestroy(*this));
  }
}

template <typename EventType>
CudaEvent::operator EventType() const {
  static_assert(std::is_same_v<EventType, CUevent>,
                "EventType must be CUevent");
  return reinterpret_cast<CUevent>(event_);
}

void CudaEvent::Synchronize() const {
  assert(event_ != nullptr);
  LC0EX_CUDA_CHECK(cuEventSynchronize(*this));
}

// Parse a buffer description from the lc0ex protobuf and create a BufferInfo
// structure that contains implicit stride information if Triton omits it.
BufferInfo MakeBufferInfo(const pblczero::Buffer& buffer) {
  absl::InlinedVector<BufferInfo::ShapeAndStride, 2> shape(
      buffer.shape().begin(), buffer.shape().end());
  if (shape.empty()) {
    throw Exception("Buffer shape is empty for " + std::string(buffer.name()) +
                    ".");
  }
  if (buffer.has_layout() && buffer.layout().strides_size() > 0) {
    auto& stride = buffer.layout().strides();
    if (stride.size() != shape.size()) {
      throw Exception(
          "Buffer layout strides size does not match shape size for " +
          std::string(buffer.name()) + ".");
    }
    std::transform(shape.begin(), shape.end(), stride.begin(), shape.begin(),
                   [](BufferInfo::ShapeAndStride s, std::int64_t stride) {
                     s.stride = stride;
                     return s;
                   });
  } else {
    std::transform(shape.rbegin(), shape.rend(), shape.rbegin(),
                   [stride = 1](BufferInfo::ShapeAndStride s) mutable {
                     s.stride = stride;
                     stride *= static_cast<std::int64_t>(s.shape);
                     return s;
                   });
  }
  size_t size_bytes =
      shape[0].shape * shape[0].stride * ElementSize(buffer.data_type());
  return {
      buffer.data_type(),
      std::move(shape),
      size_bytes,
      static_cast<ptrdiff_t>(buffer.offset()),
  };
}

// Aligned device memory allocation helper.
std::pair<CUdeviceptr, CUdeviceptr> AllocateDeviceMemory(
    std::uint64_t size_bytes, std::uint64_t alignment_bytes) {
  const auto allocation_size = size_bytes + alignment_bytes - 1;

  CUdeviceptr base = 0;
  LC0EX_CUDA_CHECK(
      cuMemAlloc(&base, static_cast<std::size_t>(allocation_size)));

  const auto base_address = static_cast<std::uint64_t>(base);
  const auto aligned_address =
      (base_address + alignment_bytes - 1) & ~(alignment_bytes - 1);
  return {base, static_cast<CUdeviceptr>(aligned_address)};
}

// The KernelNode class implementation. It represents a kernel node in the lc0ex
// graph and provides methods to launch the kernel with the appropriate
// arguments and configuration.
KernelNode::KernelNode(const pblczero::Node& source, CudaExecutable& executable,
                       const pblczero::Kernel& kernel, void* function)
    : NodeBase(source),
      function_(function),
      dynamic_shared_memory_bytes_(source.dynamic_shared_memory_bytes()) {
  ExecutableImpl& impl = static_cast<ExecutableImpl&>(executable);
  auto shared_memory = source.dynamic_shared_memory_bytes();
  if (shared_memory != 0) {
    LC0EX_CUDA_CHECK(cuFuncSetAttribute(
        reinterpret_cast<CUfunction>(function_),
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_memory));
  }
  std::copy(source.grid().begin(), source.grid().end(), grid_.begin());
  std::copy(source.block().begin(), source.block().end(), block_.begin());
  arguments_.reserve(kernel.parameters_size());
  auto argument_iter = source.arguments().begin();
  for (const auto& type : kernel.parameters()) {
    switch (type) {
      case pblczero::ParameterType_PARAMETER_TYPE_POINTER:
      case pblczero::ParameterType_PARAMETER_TYPE_U32: {
        if (argument_iter == source.arguments().end()) {
          throw Exception(
              "Kernel parameters size does not match node arguments size.");
        }
        auto& argument = *argument_iter++;
        if (argument.has_symbol()) {
          CUdeviceptr symbol = 0;
          size_t symbol_size = 0;
          std::string name(argument.symbol().symbol_name());
          CUmodule module = impl.GetBinary(argument.symbol().binary_idx());
          LC0EX_CUDA_CHECK(cuModuleGetGlobal(&symbol, &symbol_size,
                                             reinterpret_cast<CUmodule>(module),
                                             name.c_str()));
          arguments_.emplace_back(std::in_place_type<ArgumentSymbol>,
                                  reinterpret_cast<void*>(symbol));
        } else if (argument.has_allocation()) {
          switch (argument.allocation().kind()) {
            case pblczero::Node::Argument::AllocationLocation::
                ALLOCATION_PERSISTENT:
              arguments_.emplace_back(
                  std::in_place_type<ArgumentPersistentBuffer>,
                  argument.allocation().offset());
              break;
            case pblczero::Node::Argument::AllocationLocation::
                ALLOCATION_EXECUTION:
              arguments_.emplace_back(
                  std::in_place_type<ArgumentExecutionBuffer>,
                  argument.allocation().offset());
              break;
            case pblczero::Node::Argument::AllocationLocation::
                ALLOCATION_UNKNOWN:
              throw Exception("Kernel argument allocation kind is unknown.");
          }
        } else {
          throw Exception("Kernel argument is not a symbol, allocation.");
        }

        break;
      }
      case pblczero::ParameterType_PARAMETER_TYPE_NULL_POINTER:
        arguments_.emplace_back(std::in_place_type<ArgumentNull>);
        break;
      case pblczero::ParameterType_PARAMETER_TYPE_UNKNOWN:
        throw Exception("Kernel argument type is unknown.");
        break;
    }
  }
}

template <typename State>
auto KernelNode::operator()(State& state) const {
  state.argval.resize(arguments_.size());
  state.argptr.resize(arguments_.size());
  for (size_t i = 0; i < arguments_.size(); ++i) {
    std::visit(
        [&state, i](auto&& arg) {
          arg(state.argptr[i], state.argval[i], state);
        },
        arguments_[i]);
  }

  if constexpr (!State::is_cuda_capturing) {
    LC0EX_CUDA_CHECK(cuLaunchKernel(
        reinterpret_cast<CUfunction>(function_), grid_[0], grid_[1], grid_[2],
        block_[0], block_[1], block_[2], dynamic_shared_memory_bytes_,
        state.cs_.stream_, state.argptr.data(), nullptr));

    return;
  } else {
    CUgraphNode node = nullptr;
    CUDA_KERNEL_NODE_PARAMS params{};
    params.func = reinterpret_cast<CUfunction>(function_);
    params.gridDimX = grid_[0];
    params.gridDimY = grid_[1];
    params.gridDimZ = grid_[2];
    params.blockDimX = block_[0];
    params.blockDimY = block_[1];
    params.blockDimZ = block_[2];
    params.sharedMemBytes = dynamic_shared_memory_bytes_;
    params.kernelParams = state.argptr.data();

    LC0EX_CUDA_CHECK(cuGraphAddKernelNode(&node, state.graph_,
                                          state.dependencies_.data(),
                                          state.dependencies_.size(), &params));

    CUlaunchAttributeValue priority{};
    priority.priority = priority_;
    LC0EX_CUDA_CHECK(cuGraphKernelNodeSetAttribute(
        node, CU_LAUNCH_ATTRIBUTE_PRIORITY, &priority));
    return node;
  }
}

// The EventRecordNode and EventWaitNode classes implementation. They represent
// event record and wait nodes in the lc0ex graph and provide methods to record
// and wait for events on the CUDA stream.
template <RecordEventType event>
EventRecordNode<event>::EventRecordNode(const pblczero::Node& source)
    : NodeBase(source) {
  if (!source.has_record_event()) {
    throw Exception("Event record node does not have record event.");
  }
}

template <RecordEventType event>
template <typename State>
auto EventRecordNode<event>::operator()(State& state) const {
  CudaEvent* e;
  switch (event) {
    case RecordEventType::kSleep:
      e = &state.cs_.sleep_event_;
      break;
    case RecordEventType::kComputeOrdering:
      e = &state.compute_ordering_event_;
      break;
    case RecordEventType::kWdlDownloadDone:
      e = &state.cs_.wdl_download_done_;
      break;
    case RecordEventType::kMlhDownloadDone:
      e = &state.cs_.mlh_download_done_;
      break;
    case RecordEventType::kPolicyDownloadDone:
      e = &state.cs_.policy_download_done_;
      break;
  }
  if constexpr (!State::is_cuda_capturing) {
    state.cs_.stream_.RecordEvent(*e);
    return;
  } else {
    CUgraphNode node = nullptr;
    LC0EX_CUDA_CHECK(cuGraphAddEventRecordNode(&node, state.graph_,
                                               state.dependencies_.data(),
                                               state.dependencies_.size(), *e));
    return node;
  }
}

template <WaitEventType event>
EventWaitNode<event>::EventWaitNode(const pblczero::Node& source)
    : NodeBase(source) {
  if (!source.has_wait_event()) {
    throw Exception("Event wait node does not have wait event.");
  }
}

template <WaitEventType event>
template <typename State>
auto EventWaitNode<event>::operator()(State& state) const {
  CudaEvent* e;
  switch (event) {
    case WaitEventType::kComputeOrdering:
      e = &state.compute_ordering_event_;
      break;
  }
  if constexpr (!State::is_cuda_capturing) {
    state.cs_.stream_.WaitEvent(*e);
    return;
  } else {
    CUgraphNode node = nullptr;
    LC0EX_CUDA_CHECK(cuGraphAddEventWaitNode(&node, state.graph_,
                                             state.dependencies_.data(),
                                             state.dependencies_.size(), *e));
    return node;
  }
}

// The MemcpyNode class implementation. It represents a memcpy node in the lc0ex
// graph and provides methods to perform host-to-device or device-to-host memory
// copies.
template <MemcpyBuffer kind>
MemcpyNode<kind>::MemcpyNode(const pblczero::Node& source)
    : NodeBase(source), gpu_offset_{source.memcpy().gpu_offset()} {
  if (!source.has_memcpy()) {
    throw Exception("Memcpy node does not have memcpy.");
  }
}

template <MemcpyBuffer kind>
template <typename State>
auto MemcpyNode<kind>::operator()(State& state) const {
  using ComputeType = typename State::ComputeType;
  bool is_h2d = false;
  size_t bytes = 0;
  void* host_ptr = nullptr;
  CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(
      state.cs_.device_memory_.Data() + gpu_offset_);

  constexpr size_t kPolicySize = 1858;
  constexpr size_t kWdlSize = 3;
  constexpr size_t kMlhSize = 1;
  switch (kind) {
    case MemcpyBuffer::kInputMask:
      host_ptr = state.cs_.input_mask.data();
      is_h2d = true;
      bytes = state.batch_size * sizeof(uint64_t) * kInputPlanes;
      break;
    case MemcpyBuffer::kInputValue:
      host_ptr = state.cs_.input_value.data();
      is_h2d = true;
      bytes = state.batch_size * sizeof(ComputeType) * kInputPlanes;
      break;
    case MemcpyBuffer::kOutputsPolicy:
      host_ptr = state.cs_.output_policy.data();
      is_h2d = false;
      bytes = state.batch_size * sizeof(ComputeType) * kPolicySize;
      break;
    case MemcpyBuffer::kOutputsWdl:
      host_ptr = state.cs_.output_wdl.data();
      is_h2d = false;
      bytes = state.batch_size * sizeof(ComputeType) * kWdlSize;
      break;
    case MemcpyBuffer::kOutputsMlh:
      host_ptr = state.cs_.output_mlh.data();
      is_h2d = false;
      bytes = state.batch_size * sizeof(ComputeType) * kMlhSize;
      break;
  }

  CudaBuffer<ComputeType> device_buffer =
      state.cs_.device_memory_.template AsSpan<ComputeType>(gpu_offset_, bytes);
  if constexpr (!State::is_cuda_capturing) {
    if (is_h2d) {
      LC0EX_CUDA_CHECK(
          cuMemcpyHtoDAsync(device_ptr, host_ptr, bytes, state.cs_.stream_));
    } else {
      LC0EX_CUDA_CHECK(
          cuMemcpyDtoHAsync(host_ptr, device_ptr, bytes, state.cs_.stream_));
    }
    return;
  } else {
    CUgraphNode node = nullptr;
    CUDA_MEMCPY3D copy_params{};
    copy_params.srcMemoryType =
        is_h2d ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
    copy_params.srcHost = is_h2d ? host_ptr : nullptr;
    copy_params.srcDevice = is_h2d ? 0 : device_ptr;
    copy_params.dstMemoryType =
        is_h2d ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST;
    copy_params.dstHost = is_h2d ? 0 : host_ptr;
    copy_params.dstDevice = is_h2d ? device_ptr : 0;
    copy_params.WidthInBytes = bytes;
    copy_params.Height = 1;
    copy_params.Depth = 1;
    LC0EX_CUDA_CHECK(cuGraphAddMemcpyNode(
        &node, state.graph_, state.dependencies_.data(),
        state.dependencies_.size(), &copy_params, nullptr));
    return node;
  }
}

// The CudaProgram class implementation. It represents a program in the lc0ex
// executable and provides methods to launch the graph.
CudaProgram::CudaProgram(const pblczero::Program& source,
                         CudaExecutable& executable,
                         const std::vector<pblczero::Kernel>& kernels) {
  LCTRACE_FUNCTION_SCOPE;
  using NodeVector = decltype(nodes_);
  pblczero::ProgramMetadata metadata;
  metadata.ParseFromString(source.metadata());
  batch_size_ = metadata.batch_size();
  ExecutableImpl& impl = static_cast<ExecutableImpl&>(executable);

  if (source.has_execution_allocation()) {
    impl.BuildExecutionAllocation(source);
  }

  nodes_.reserve(source.nodes_size());
  size_t idx = 0;
  for (const auto& node : source.nodes()) {
    size_t i = idx++;
    if (node.has_kernel_idx()) {
      if (node.kernel_idx() >= kernels.size()) {
        throw Exception("Node (" + std::to_string(i) +
                        ") kernel index is out of range in " +
                        std::string(source.name()) + ".");
      }
      auto& kernel = kernels[node.kernel_idx()];
      CUmodule module = impl.GetBinary(kernel.binary_idx());
      std::string kernel_name(kernel.function());
      CUfunction function = nullptr;
      LC0EX_CUDA_CHECK(
          cuModuleGetFunction(&function, module, kernel_name.c_str()));
      if (node.grid_size() != 3) {
        throw Exception("Node (" + std::to_string(i) +
                        ") grid size is not 3 in " +
                        std::string(source.name()) + ".");
      }
      if (node.block_size() != 3) {
        throw Exception("Node (" + std::to_string(i) +
                        ") block size is not 3 in " +
                        std::string(source.name()) + ".");
      }
      if (!node.has_dynamic_shared_memory_bytes()) {
        throw Exception("Node (" + std::to_string(i) +
                        ") dynamic shared memory bytes is not set in " +
                        std::string(source.name()) + ".");
      }
      nodes_.emplace_back(std::in_place_type<KernelNode>, node, executable,
                          kernel, reinterpret_cast<void*>(function));
      std::get<KernelNode>(nodes_.back()).SetDebugState(node.kernel_idx(), i);
    } else if (node.has_record_event()) {
      static const std::map<
          std::string_view,
          std::function<void(NodeVector&, const pblczero::Node&)>>
          event_node_creators = {
              {"/event/sleep",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kSleep>>,
                     node);
               }},
              {"/event/compute_ordering",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kComputeOrdering>>,
                     node);
               }},
              {"/event/wdl_done",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kWdlDownloadDone>>,
                     node);
               }},
              {"/event/mlh_done",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kMlhDownloadDone>>,
                     node);
               }},
              {"/event/policy_done",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventRecordNode<RecordEventType::kPolicyDownloadDone>>,
                     node);
               }}};
      auto it = event_node_creators.find(node.record_event());
      if (it == event_node_creators.end()) {
        throw Exception("Unknown record event name (" +
                        std::string(node.record_event()) + ") in node (" +
                        std::to_string(i) + ") in " +
                        std::string(source.name()) + ".");
      }
      it->second(nodes_, node);
    } else if (node.has_wait_event()) {
      static const std::map<
          std::string_view,
          std::function<void(NodeVector&, const pblczero::Node&)>>
          wait_node_creators = {
              {"/event/compute_ordering",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         EventWaitNode<WaitEventType::kComputeOrdering>>,
                     node);
               }}};
      auto it = wait_node_creators.find(node.wait_event());
      if (it == wait_node_creators.end()) {
        throw Exception("Unknown wait event name (" +
                        std::string(node.wait_event()) + ") in node (" +
                        std::to_string(i) + ") in " +
                        std::string(source.name()) + ".");
      }
      it->second(nodes_, node);
    } else if (node.has_memcpy()) {
      static const std::map<
          std::string_view,
          std::function<void(NodeVector&, const pblczero::Node&)>>
          memcpy_node_creators = {
              {"/input/plane_masks",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kInputMask>>,
                     node);
               }},
              {"/input/plane_values",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kInputValue>>,
                     node);
               }},
              {"/output/policy",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<
                         MemcpyNode<MemcpyBuffer::kOutputsPolicy>>,
                     node);
               }},
              {"/output/wdl",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kOutputsWdl>>,
                     node);
               }},
              {"/output/mlh",
               [](NodeVector& nodes, const pblczero::Node& node) {
                 nodes.emplace_back(
                     std::in_place_type<MemcpyNode<MemcpyBuffer::kOutputsMlh>>,
                     node);
               }}};
      auto it = memcpy_node_creators.find(node.memcpy().name());
      if (it == memcpy_node_creators.end()) {
        throw Exception("Unknown memcpy buffer name (" +
                        std::string(node.memcpy().name()) + ") in node (" +
                        std::to_string(i) + ") in " +
                        std::string(source.name()) + ".");
      }
      it->second(nodes_, node);
    } else {
      throw Exception("Unknown node type (" + std::to_string(i) + ") in " +
                      std::string(source.name()) + ".");
    }
  }
}

template <typename State>
void CudaProgram::Run(State& state) const {
  for (const auto& node : nodes_) {
    std::visit([&](auto&& arg) { arg(state); }, node);
  }
}

GraphCapture::GraphCapture() {
  CUgraph graph = nullptr;
  LC0EX_CUDA_CHECK(cuGraphCreate(&graph, 0));
  graph_ = reinterpret_cast<Graph>(graph);
}

GraphCapture::GraphCapture(GraphCapture&& other) noexcept
    : graph_(other.graph_) {
  other.graph_ = nullptr;
}

GraphCapture& GraphCapture::operator=(GraphCapture&& other) noexcept {
  if (this != &other) {
    if (graph_) {
      LC0EX_CUDA_CHECK(cuGraphDestroy(*this));
    }
    graph_ = other.graph_;
    other.graph_ = nullptr;
  }
  return *this;
}

GraphCapture::~GraphCapture() {
  if (graph_) {
    LC0EX_CUDA_CHECK(cuGraphDestroy(*this));
  }
}

template <typename GraphType>
GraphCapture::operator GraphType() const {
  static_assert(std::is_same_v<GraphType, CUgraph>,
                "GraphType must be CUgraph");
  return reinterpret_cast<GraphType>(graph_);
}

template <typename State>
GraphCapture CudaProgram::Capture(GraphMode mode, State& state) const {
  state.graph_nodes_.reserve(nodes_.size());
  for (const auto& node : nodes_) {
    state.dependencies_.clear();
    if (mode == GraphMode::kDag) {
      auto deps =
          std::visit([&](auto&& arg) { return arg.GetDependecies(); }, node);
      for (auto dep : deps) {
        state.dependencies_.push_back(state.graph_nodes_[dep]);
      }
    } else {
      state.dependencies_.push_back(state.graph_nodes_.back());
    }
    CUgraphNode graph_node =
        std::visit([&](auto&& arg) { return arg(state); }, node);
    state.graph_nodes_.push_back(graph_node);
  }
  GraphCapture rv{std::move(state.graph_)};
  return rv;
}

namespace {

[[maybe_unused]]
inline CUdevice& ToDevice(Device& device) {
  return static_cast<CUdevice&>(device);
}
[[maybe_unused]]
inline const CUdevice& ToDevice(const Device& device) {
  return static_cast<const CUdevice&>(device);
}
[[maybe_unused]]
inline CUcontext& ToContext(Context& context) {
  return reinterpret_cast<CUcontext&>(context);
}
[[maybe_unused]]
const CUcontext& ToContext(const Context& context) {
  return reinterpret_cast<const CUcontext&>(context);
}

}  // namespace

// The CudaMemory class implementation. It represents a block of device memory
// and provides methods to manage its lifecycle and operations.
CudaMemory::CudaMemory(size_t size, size_t alignment)
    : MemoryBase(size, alignment) {
  assert(size > 0);
  assert(alignment > 0);
  assert(std::has_single_bit(alignment));
  std::tie(reinterpret_cast<CUdeviceptr&>(device_ptr_),
           reinterpret_cast<CUdeviceptr&>(aligned_ptr_)) =
      AllocateDeviceMemory(size, alignment);
}

CudaMemory::CudaMemory(CudaMemory&& other) noexcept
    : MemoryBase(std::move(other)), device_ptr_(other.device_ptr_) {
  other.device_ptr_ = 0;
}

CudaMemory& CudaMemory::operator=(CudaMemory&& other) noexcept {
  if (this != &other) {
    if (device_ptr_ != 0) {
      LC0EX_CUDA_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(device_ptr_)));
    }
    MemoryBase::operator=(std::move(other));
    device_ptr_ = other.device_ptr_;
    other.device_ptr_ = 0;
  }
  return *this;
}

CudaMemory::~CudaMemory() {
  if (device_ptr_ != 0) {
    LC0EX_CUDA_CHECK(cuMemFree(reinterpret_cast<CUdeviceptr>(device_ptr_)));
  }
}

// The CudaHostMemory class implementation. It represents a block of pinned host
// memory and provides methods to manage its lifecycle and operations.
CudaHostMemory::CudaHostMemory(size_t size, size_t alignment)
    : MemoryBase(size, alignment) {
  assert(size > 0);
  assert(alignment > 0);
  assert(std::has_single_bit(alignment));
  assert(size % alignment == 0);
  void* host_ptr = nullptr;
  size = (size + alignment - 1) & ~(alignment - 1);
  LC0EX_CUDA_CHECK(cuMemHostAlloc(&host_ptr, size, CU_MEMHOSTALLOC_PORTABLE));
  const auto base_address = reinterpret_cast<uintptr_t>(host_ptr);
  const auto aligned_address =
      (base_address + alignment - 1) & ~(alignment - 1);
  host_ptr_ = host_ptr;
  aligned_ptr_ = reinterpret_cast<char*>(aligned_address);
}

CudaHostMemory::CudaHostMemory(CudaHostMemory&& other) noexcept
    : MemoryBase(std::move(other)), host_ptr_(other.host_ptr_) {
  other.host_ptr_ = nullptr;
}

CudaHostMemory::~CudaHostMemory() {
  if (host_ptr_ != nullptr) {
    LC0EX_CUDA_CHECK(cuMemFreeHost(host_ptr_));
  }
}

// The CudaGraphExec class implementation. It represents a CUDA graph execution
// object and provides methods to manage its lifecycle and operations.
CudaGraphExec::CudaGraphExec(const GraphCapture& graph) {
  CUgraphExec graph_exec = nullptr;
  LC0EX_CUDA_CHECK(cuGraphInstantiate(
      &graph_exec, graph, CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY));
  graph_exec_ = reinterpret_cast<GraphExec>(graph_exec);
}

void CudaGraphExec::Upload(CudaStream& stream) const {
  LC0EX_CUDA_CHECK(
      cuGraphUpload(reinterpret_cast<CUgraphExec>(graph_exec_), stream));
}

CudaGraphExec::CudaGraphExec(CudaGraphExec&& other) noexcept
    : graph_exec_(other.graph_exec_) {
  other.graph_exec_ = nullptr;
}

CudaGraphExec& CudaGraphExec::operator=(CudaGraphExec&& other) noexcept {
  if (this != &other) {
    if (graph_exec_) {
      LC0EX_CUDA_CHECK(cuGraphExecDestroy(*this));
    }
    graph_exec_ = other.graph_exec_;
    other.graph_exec_ = nullptr;
  }
  return *this;
}

CudaGraphExec::~CudaGraphExec() {
  if (graph_exec_) {
    LC0EX_CUDA_CHECK(cuGraphExecDestroy(*this));
  }
}

template <typename GraphExecType>
CudaGraphExec::operator GraphExecType() const {
  return reinterpret_cast<GraphExecType>(graph_exec_);
}

void CudaGraphExec::Launch(CudaStream& stream) const {
  LC0EX_CUDA_CHECK(
      cuGraphLaunch(reinterpret_cast<CUgraphExec>(graph_exec_), stream));
}

// The CudaRuntime class implementation. It represents a CUDA  context which
// uses primary context to allow interoperability with other CUDA APIs and
// libraries.
CudaRuntime::CudaRuntime(int device_ordinal) {
  LCTRACE_FUNCTION_SCOPE;
  LC0EX_CUDA_CHECK(cuInit(0));
  LC0EX_CUDA_CHECK(cuDeviceGet(&ToDevice(device_), device_ordinal));
  LC0EX_CUDA_CHECK(cuDevicePrimaryCtxRetain(&ToContext(context_), device_));
  SetCurrent();
}

CudaRuntime::~CudaRuntime() {
  if (context_) LC0EX_CUDA_CHECK(cuDevicePrimaryCtxRelease(ToDevice(device_)));
}

void CudaRuntime::SetCurrent() const {
  LC0EX_CUDA_CHECK(cuCtxSetCurrent(ToContext(context_)));
}

// The CudaExecutable class implementation. It implements and NeuralExecutable
// which can have many programs using shared kernels and buffers. It provides
// methods to run or capture graphs.
CudaExecutable::CudaExecutable(const pblczero::NeuralExecutable& source) {
  if (source.magic() != kMagic) throw Exception("Invalid lc0ex magic.");
  if (source.format() != kFormat) {
    throw Exception("Unsupported lc0ex format generation.");
  }

  ExecutableImpl* impl = static_cast<ExecutableImpl*>(this);

  impl->BuildModules(source);
  impl->BuildPersistentAllocation(source);
  impl->BuildExecutionAllocation(source);
  impl->BuildPrograms(source);
}

size_t CudaExecutable::GetPersistentAllocationSize() const {
  return persistent_allocation_.Size();
}

size_t CudaExecutable::GetExecutionAllocationSize() const {
  return execution_allocation_.size_bytes_;
}

size_t CudaExecutable::GetExecutionAllocationAlignment() const {
  return execution_allocation_.alignment_bytes_;
}

size_t CudaExecutable::GetMaxBatchSize() const {
  return programs_.back().Size();
}

void CudaExecutable::CopyPersistentFromHost(
    std::span<const std::byte> source,
    std::optional<std::size_t> size_bytes) const {
  const std::size_t copy_size = size_bytes.value_or(source.size());
  if (copy_size == 0 || !persistent_allocation_) return;

  LC0EX_CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)persistent_allocation_.Data(),
                                source.data(), copy_size));
}

template <typename T>
void CudaExecutable::Run(size_t batch_size,
                         ComputationState<CudaRuntime, T>& cs,
                         CudaEvent& compute_ordering_event) {
  auto program =
      std::lower_bound(programs_.begin(), programs_.end(), batch_size,
                       [](const CudaProgram& program, size_t batch_size) {
                         return program.Size() < batch_size;
                       });
  if (program == programs_.end()) {
    throw Exception("Batch size exceeds maximum supported by executable.");
  }
  LaunchState<T> state{
      cs, batch_size, persistent_allocation_, compute_ordering_event, {}, {}};
  program->Run(state);
}

template void CudaExecutable::Run<float>(
    size_t batch_size, ComputationState<CudaRuntime, float>& state,
    CudaEvent& compute_ordering_event);
template void CudaExecutable::Run<_Float16>(
    size_t batch_size, ComputationState<CudaRuntime, _Float16>& state,
    CudaEvent& compute_ordering_event);

template <typename T>
GraphCapture CudaExecutable::Capture(GraphMode mode, size_t batch_size,
                                     ComputationState<CudaRuntime, T>& cs,
                                     CudaEvent& compute_ordering_event) {
  auto program =
      std::lower_bound(programs_.begin(), programs_.end(), batch_size,
                       [](const CudaProgram& program, size_t batch_size) {
                         return program.Size() < batch_size;
                       });
  if (program == programs_.end()) {
    throw Exception("Batch size exceeds maximum supported by executable.");
  }
  CaptureState<T> state{
      {cs, batch_size, persistent_allocation_, compute_ordering_event, {}, {}},
      {},
      {},
      {}};
  return program->Capture(mode, state);
}

template GraphCapture CudaExecutable::Capture<float>(
    GraphMode mode, size_t batch_size,
    ComputationState<CudaRuntime, float>& state,
    CudaEvent& compute_ordering_event);
template GraphCapture CudaExecutable::Capture<_Float16>(
    GraphMode mode, size_t batch_size,
    ComputationState<CudaRuntime, _Float16>& state,
    CudaEvent& compute_ordering_event);

}  // namespace lczero::lc0ex::cuda
