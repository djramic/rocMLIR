//===-- Shared/APITypes.h - Offload and plugin API types --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines types used in the interface between the user code, the
// target independent offload runtime library, and target dependent plugins.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_API_TYPES_H
#define OMPTARGET_SHARED_API_TYPES_H

#include "Environment.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <variant>

#ifdef OMPT_SUPPORT
#include <omp-tools.h>
#endif

extern "C" {

/// This struct is a record of an entry point or global. For a function
/// entry point the size is expected to be zero
struct __tgt_offload_entry {
  void *addr;       // Pointer to the offload entry info (function or global)
  char *name;       // Name of the function or global
  size_t size;      // Size of the entry info (0 if it is a function)
  int32_t flags;    // Flags associated with the entry, e.g. 'link'.
  int32_t data;     // Extra data associated with the entry.
};

/// This struct is a record of the device image information
struct __tgt_device_image {
  void *ImageStart;                  // Pointer to the target code start
  void *ImageEnd;                    // Pointer to the target code end
  __tgt_offload_entry *EntriesBegin; // Begin of table with all target entries
  __tgt_offload_entry *EntriesEnd;   // End of table (non inclusive)
};

struct __tgt_device_info {
  void *Context = nullptr;
  void *Device = nullptr;
};

/// This struct is a record of all the host code that may be offloaded to a
/// target.
struct __tgt_bin_desc {
  int32_t NumDeviceImages;          // Number of device types supported
  __tgt_device_image *DeviceImages; // Array of device images (1 per dev. type)
  __tgt_offload_entry *HostEntriesBegin; // Begin of table with all host entries
  __tgt_offload_entry *HostEntriesEnd;   // End of table (non inclusive)
};

/// This struct contains the offload entries identified by the target runtime
struct __tgt_target_table {
  __tgt_offload_entry *EntriesBegin; // Begin of the table with all the entries
  __tgt_offload_entry
      *EntriesEnd; // End of the table with all the entries (non inclusive)
};

/// This struct contains a handle to a loaded binary in the plugin device.
struct __tgt_device_binary {
  uintptr_t handle;
};

// clang-format on

// OMPT: Forward declare for function pointer type
namespace llvm::omp::target::ompt {
struct OmptEventInfoTy;
} // namespace llvm::omp::target::ompt

/// This struct contains information exchanged between different asynchronous
/// operations for device-dependent optimization and potential synchronization
struct __tgt_async_info {
  // A pointer to a queue-like structure where offloading operations are issued.
  // We assume to use this structure to do synchronization. In CUDA backend, it
  // is CUstream.
  void *Queue = nullptr;

  /// A collection of allocations that are associated with this stream and that
  /// should be freed after finalization.
  llvm::SmallVector<void *, 2> AssociatedAllocations;

  /// The kernel launch environment used to issue a kernel. Stored here to
  /// ensure it is a valid location while the transfer to the device is
  /// happening.
  KernelLaunchEnvironmentTy KernelLaunchEnvironment;

  /// Use for sync interface. When false => synchronous execution
  bool ExecAsync = true;
  /// Maintain the actal data for the OMPT stuff
  llvm::omp::target::ompt::OmptEventInfoTy *OmptEventInfo = nullptr;
};

/// This struct contains all of the arguments to a target kernel region launch.
struct KernelArgsTy {
  uint32_t Version = 0; // Version of this struct for ABI compatibility.
  uint32_t NumArgs = 0; // Number of arguments in each input pointer.
  void **ArgBasePtrs =
      nullptr;                 // Base pointer of each argument (e.g. a struct).
  void **ArgPtrs = nullptr;    // Pointer to the argument data.
  int64_t *ArgSizes = nullptr; // Size of the argument data in bytes.
  int64_t *ArgTypes = nullptr; // Type of the data (e.g. to / from).
  void **ArgNames = nullptr;   // Name of the data for debugging, possibly null.
  void **ArgMappers = nullptr; // User-defined mappers, possibly null.
  uint64_t Tripcount =
      0; // Tripcount for the teams / distribute loop, 0 otherwise.
  struct {
    uint64_t NoWait : 1; // Was this kernel spawned with a `nowait` clause.
    uint64_t IsCUDA : 1; // Was this kernel spawned via CUDA.
    uint64_t Unused : 62;
  } Flags = {0, 0, 0};
  // The number of teams (for x,y,z dimension).
  uint32_t NumTeams[3] = {0, 0, 0};
   // The number of threads (for x,y,z dimension).
  uint32_t ThreadLimit[3] = {0, 0, 0};
  uint32_t DynCGroupMem = 0;     // Amount of dynamic cgroup memory requested.
};
static_assert(sizeof(KernelArgsTy().Flags) == sizeof(uint64_t),
              "Invalid struct size");
static_assert(sizeof(KernelArgsTy) ==
                  (8 * sizeof(int32_t) + 3 * sizeof(int64_t) +
                   4 * sizeof(void **) + 2 * sizeof(int64_t *)),
              "Invalid struct size");

/// Flat array of kernel launch parameters and their total size.
struct KernelLaunchParamsTy {
  /// Size of the Data array.
  size_t Size = 0;
  /// Flat array of kernel parameters.
  void *Data = nullptr;
  /// Ptrs to the Data entries. Only strictly required for the host plugin.
  void **Ptrs = nullptr;
};
}

#endif // OMPTARGET_SHARED_API_TYPES_H
