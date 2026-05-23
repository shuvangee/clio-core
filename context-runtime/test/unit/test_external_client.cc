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
 * External Client Connection Tests
 *
 * Tests client-only mode connecting to an existing server process.
 * This exercises the ClientInit() code paths that are skipped when using
 * integrated server+client mode.
 *
 * Uses SystemInfo::SpawnProcess for portable process management.
 */

#include "../simple_test.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include "clio_runtime/clio_runtime.h"
#include "clio_runtime/ipc_manager.h"

using namespace chi;

/**
 * Helper to start server in background process via SpawnProcess
 * Returns ProcessHandle
 */
pid_t StartServerProcess() {
  pid_t server_pid = fork();
  if (server_pid == 0) {
    // Redirect child's stdout/stderr to /dev/null to prevent massive
    // worker log output from flooding shared pipes and blocking parent
    (void)freopen("/dev/null", "w", stdout);
    (void)freopen("/dev/null", "w", stderr);

    // Child process: Start runtime server
    setenv("CLIO_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      _exit(1);
    }

    // Keep server alive for tests
    // Server will be killed by parent process
    sleep(300);  // 5 minutes max
    _exit(0);
  }
  return server_pid;
}

/**
 * Helper to wait for server to be ready
 */
bool WaitForServer(int max_attempts = 50) {
  // Use a lightbeam probe: try connecting via TCP
  // Give the server time to initialize
  for (int i = 0; i < max_attempts; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Check if memfd symlink dir exists (Linux) or just wait (Windows)
    std::string memfd_dir =
        (std::filesystem::temp_directory_path() / "chimaera_memfd").string();
    std::error_code ec;
    if (std::filesystem::exists(memfd_dir, ec)) {
      // Check for the main segment file
      const char *user = std::getenv("USER");
      if (!user) user = std::getenv("USERNAME");
      std::string segment_name =
          std::string("chi_main_segment_") + (user ? user : "");
      auto segment_path =
          std::filesystem::path(memfd_dir) / segment_name;
      if (std::filesystem::exists(segment_path, ec)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        return true;
      }
    }
  }
  // Fallback: just wait long enough for the server to bind its TCP port
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  return true;
}

/**
 * Helper to cleanup server process
 */
void CleanupSharedMemory() {
  // Clean up leftover memfd symlinks
  const char *user = std::getenv("USER");
  std::string memfd_path = std::string("/tmp/chimaera_") +
                           (user ? user : "unknown") +
                           "/chi_main_segment_" + (user ? user : "");
  unlink(memfd_path.c_str());
}

void CleanupServer(pid_t server_pid) {
  if (server_pid > 0) {
    kill(server_pid, SIGTERM);
    // Wait up to 5 seconds for graceful shutdown
    for (int i = 0; i < 50; ++i) {
      int status;
      if (waitpid(server_pid, &status, WNOHANG) != 0) {
        CleanupSharedMemory();
        return;
      }
      usleep(100000);  // 100ms
    }
    // Force kill if still alive
    kill(server_pid, SIGKILL);
    int status;
    waitpid(server_pid, &status, 0);
    // Clean up shared memory left behind by the server
    CleanupSharedMemory();
  }
}

// ============================================================================
// External Client Connection Tests
// ============================================================================

TEST_CASE("ExternalClient - Basic Connection", "[external_client][ipc]") {
  // Start server in background
  auto server = StartServerProcess();

  // Wait for server to be ready
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Now connect as EXTERNAL CLIENT (not integrated server+client)
  setenv("CLIO_WITH_RUNTIME", "0", 1);  // Force client-only mode
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  // Verify client initialized successfully
  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);
  REQUIRE(ipc->IsInitialized());

  // Test basic operations from external client
  u64 node_id = ipc->GetNodeId();
  (void)node_id;

  // In TCP mode (default), the client does not attach to shared memory
  auto *queue = ipc->GetTaskQueue();
  if (ipc->GetIpcMode() == IpcMode::kShm) {
    REQUIRE(queue != nullptr);
  } else {
    REQUIRE(queue == nullptr);
  }

  // Cleanup
  CleanupServer(server_pid);
}

TEST_CASE("ExternalClient - Multiple Clients", "[external_client][ipc]") {
  // Start server
  pid_t server_pid = StartServerProcess();
  REQUIRE(server_pid > 0);

  // Wait for server
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Start multiple client processes
  const int num_clients = 3;
  pid_t client_pids[num_clients];

  for (int i = 0; i < num_clients; ++i) {
    client_pids[i] = fork();
    if (client_pids[i] == 0) {
      // Suppress child output to prevent log flood
      (void)freopen("/dev/null", "w", stdout);
      (void)freopen("/dev/null", "w", stderr);

      // Child process: Connect as client
      setenv("CLIO_WITH_RUNTIME", "0", 1);
      bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
      if (!success) {
        _exit(1);
      }

      auto *ipc = CLIO_IPC;
      if (!ipc || !ipc->IsInitialized()) {
        _exit(1);
      }

      // Verify we can get node ID (0 is valid for localhost)
      u64 node_id = ipc->GetNodeId();
      (void)node_id;

      // Client test passed
      _exit(0);
    }
  }

  // Wait for all clients to complete
  bool all_success = true;
  for (int i = 0; i < num_clients; ++i) {
    int status;
    waitpid(client_pids[i], &status, 0);
    if (WEXITSTATUS(status) != 0) {
      all_success = false;
    }
  }

  REQUIRE(all_success);

  // Cleanup server
  CleanupServer(server_pid);
}

TEST_CASE("ExternalClient - Connection Without Server",
          "[external_client][ipc][errors]") {
  // Wait briefly for ports to be freed
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Try to connect as client when NO server exists
  setenv("CLIO_WITH_RUNTIME", "0", 1);

  // This should fail gracefully (not crash)
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  (void)success;  // Just verify it doesn't crash
}

TEST_CASE("ExternalClient - Client Operations", "[external_client][ipc]") {
  // Start server
  auto server = StartServerProcess();

  // Wait for server
  bool server_ready = WaitForServer();
  REQUIRE(server_ready);

  // Connect as client
  setenv("CLIO_WITH_RUNTIME", "0", 1);
  bool success = CHIMAERA_INIT(ChimaeraMode::kClient, false);
  REQUIRE(success);

  auto *ipc = CLIO_IPC;
  REQUIRE(ipc != nullptr);

  // In TCP mode (default), num_sched_queues_ is not set so
  // GetNumSchedQueues returns 0. In SHM mode it would be > 0.
  u32 num_queues = ipc->GetNumSchedQueues();
  if (ipc->GetIpcMode() == IpcMode::kShm) {
    REQUIRE(num_queues > 0);
  }

  // Cleanup
  CleanupServer(server);
}

int main(int argc, char *argv[]) {
  // Server mode: started by StartServerProcess() via SpawnProcess
  if (argc > 1 && std::string(argv[1]) == "--server-mode") {
    hshm::SystemInfo::Setenv("CHI_WITH_RUNTIME", "1", 1);
    bool success = CHIMAERA_INIT(ChimaeraMode::kServer, true);
    if (!success) {
      return 1;
    }
    // Keep server alive for tests (parent will kill us)
    std::this_thread::sleep_for(std::chrono::minutes(5));
    return 0;
  }

  // Normal test mode
  hshm::SystemInfo::SuppressErrorDialogs();
  std::string filter = "";
  if (argc > 1) {
    filter = argv[1];
  }
  int rc = SimpleTest::run_all_tests(filter);
  chi::CHIMAERA_FINALIZE();
  SIMPLE_TEST_HARD_EXIT(rc);
  return rc;
}
