/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * IPC Error Handling Tests
 *
 * Tests error conditions and failure paths in IpcManager that are not
 * exercised by happy-path tests.
 */

#include "../simple_test.h"

#include <cstdlib>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

using namespace chi;

// ============================================================================
// Global Setup - Initialize once for all tests
// ============================================================================
static bool InitializeRuntime() {
  static bool initialized = false;
  if (!initialized) {
    bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);
    initialized = success;
    if (success) SimpleTest::g_test_finalize = chi::CHIMAERA_FINALIZE;
    return success;
  }
  return true;
}

// ============================================================================
// Connection Error Tests
// ============================================================================

// NOTE: This test is disabled because CHIMAERA_INIT has a static guard
// that prevents multiple initializations in the same process. Once it
// succeeds in any test, it will return true in all subsequent tests.
// This test would need to run in a separate process to work correctly.
/*
TEST_CASE("IpcErrors - Client Connect Without Server", "[ipc][errors]") {
  // Ensure no server is running by clearing any existing IPCs
  // (This may fail if no IPCs exist, which is fine)

  // Try to connect as client when NO server exists
  setenv("CLIO_WITH_RUNTIME", "0", 1);

  // This should timeout and fail gracefully (not crash)
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(!success);

  // Verify IPC manager is not initialized
  // Note: CLIO_IPC may be null or in uninitialized state
  auto *ipc = CLIO_IPC;
  if (ipc != nullptr) {
    REQUIRE(!ipc->IsInitialized());
  }
}
*/

// NOTE: This test is disabled because it tries to call CHIMAERA_INIT
// which can only be called once per process. It would need to run in
// a separate process to work correctly.
/*
TEST_CASE("IpcErrors - Connection Timeout", "[ipc][errors]") {
  // Start a server that will immediately exit (simulating crash)
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Child: Start server then immediately exit
    setenv("CLIO_WITH_RUNTIME", "1", 1);
    CHIMAERA_INIT(ChimaeraMode::kServer, true);
    exit(0);  // Exit immediately
  }

  // Wait for server to exit
  int status;
  waitpid(server_pid, &status, 0);

  // Small delay
  usleep(100000);

  // Now try to connect - server is gone
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);

  // May succeed or fail depending on timing and leftover shm
  // The important thing is it doesn't crash
}
*/

// ============================================================================
// Memory Allocation Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Huge Buffer Allocation", "[ipc][errors][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Try to allocate impossibly large buffer
  size_t huge_size = ctp::Unit<size_t>::Terabytes(100);
  auto buf = ipc->AllocateBuffer(huge_size);
  REQUIRE(buf.IsNull());
}

TEST_CASE("IpcErrors - Zero Size Allocation", "[ipc][errors][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  auto buf = ipc->AllocateBuffer(0);
  if (!buf.IsNull()) {
    ipc->FreeBuffer(buf);
  }
}

TEST_CASE("IpcErrors - Invalid Buffer Free", "[ipc][errors][memory]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  FullPtr<char> null_buf;
  ipc->FreeBuffer(null_buf);
}

// ============================================================================
// Host/Network Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Node ID", "[ipc][errors][network]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  auto *host = ipc->GetHost(0xDEADBEEF);
  REQUIRE(host == nullptr);

  host = ipc->GetHost(0xFFFFFFFF);
  REQUIRE(host == nullptr);
}

TEST_CASE("IpcErrors - Invalid IP Address", "[ipc][errors][network]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  auto *host = ipc->GetHostByIp("999.999.999.999");
  REQUIRE(host == nullptr);

  host = ipc->GetHostByIp("");
  REQUIRE(host == nullptr);

  host = ipc->GetHostByIp("not.an.ip.address");
  REQUIRE(host == nullptr);
}

TEST_CASE("IpcErrors - Network Client Creation Failure",
          "[ipc][errors][network]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  try {
    auto *client = ipc->GetOrCreateClient("invalid://address", 0);
    (void)client;
  } catch (const std::exception &e) {
    // Exception is acceptable for invalid address
  }
}

// ============================================================================
// Queue Operation Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Network Queue Operations", "[ipc][errors][queue]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Try to pop from empty network queue across the latency/IO lanes
  Future<Task> future;
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendInLatency, future));
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendInIO, future));
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendOutLatency, future));
  REQUIRE(!ipc->TryPopNetTask(NetQueuePriority::kSendOutIO, future));

}

// ============================================================================
// Shared Memory Error Tests
// ============================================================================

TEST_CASE("IpcErrors - Invalid Allocator Registration", "[ipc][errors][shm]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // Try to register with invalid allocator ID
  ctp::ipc::AllocatorId invalid_id(0xFFFF, 0xFFFF);
  bool registered = ipc->RegisterMemory(invalid_id);
  (void)registered;
}

TEST_CASE("IpcErrors - GetClientShmInfo Invalid Index",
          "[ipc][errors][shm]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  ClientShmInfo info = ipc->GetClientShmInfo(9999);
  (void)info;
}

// ============================================================================
// Scheduler Error Tests
// ============================================================================

TEST_CASE("IpcErrors - SetNumSchedQueues Edge Cases", "[ipc][errors][sched]") {
  REQUIRE(InitializeRuntime());

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  u32 original = ipc->GetNumSchedQueues();
  REQUIRE(original > 0);

  ipc->SetNumSchedQueues(0);
  ipc->SetNumSchedQueues(1000000);
  ipc->SetNumSchedQueues(original);
}

// ============================================================================
// Multi-Process Error Tests
// ============================================================================

// NOTE: This test is disabled because it forks multiple processes that each
// try to start a full runtime (workers, modules, servers). This is very heavy
// and causes timeouts. It's better suited for integration tests.
/*
TEST_CASE("IpcErrors - Concurrent Init/Finalize", "[ipc][errors][multiproc]") {
  // Test concurrent initialization attempts
  const int num_procs = 3;
  pid_t pids[num_procs];

  for (int i = 0; i < num_procs; ++i) {
    pids[i] = fork();
    if (pids[i] == 0) {
      // Child: Try to initialize
      bool success = CHIMAERA_INIT(ChimaeraMode::kClient, true);

      if (success) {
        auto *ipc = CLIO_IPC;
        if (ipc && ipc->IsInitialized()) {
          // Do some operations
          ipc->GetNodeId();
          ipc->GetNumSchedQueues();

          // Finalize using CLIO Runtime API
          CLIO_CHIMAERA_MANAGER->ServerFinalize();
        }
        exit(0);
      } else {
        exit(1);
      }
    }
  }

  // Wait for all children
  int success_count = 0;
  for (int i = 0; i < num_procs; ++i) {
    int status;
    waitpid(pids[i], &status, 0);
    if (WEXITSTATUS(status) == 0) {
      success_count++;
    }
  }

  // At least one should succeed, others may fail due to race
  REQUIRE(success_count >= 1);
}
*/

// ============================================================================
// Global Cleanup - Finalize once at the end
// ============================================================================

TEST_CASE("IpcErrors - ZZZ Final Cleanup", "[ipc][errors][cleanup]") {
  // Finalize before hard exit to release ZMQ ports
  chi::CHIMAERA_FINALIZE();
  SIMPLE_TEST_HARD_EXIT(0);
}

SIMPLE_TEST_MAIN()
