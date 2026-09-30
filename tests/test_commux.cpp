// Multi-rank test for commux::ProcessGroupUCX. Bootstraps ranks through a
// c10d::TCPStore (rank 0 is the store server), then exercises the two
// capabilities that motivate this backend:
//
//   1. Tag-matched point-to-point: rank 1 receives tag B *before* tag A while
//      rank 0 sends A then B. Correct delivery proves real (sender,tag)
//      matching
//      -- a case NCCL gets wrong because it ignores tags and matches by order.
//      Posted non-blocking (post all, then wait) so it is deadlock-free under
//      the rendezvous protocol CUDA uses.
//   2. allreduce SUM/MIN/MAX, reduce-to-root, barrier.
//
// Usage:
//   test_commux <rank> <world_size> [master_host=127.0.0.1] [master_port=29555]
//   UCXPG_DEVICE=cuda  runs the same checks on GPU tensors.

#include <sys/resource.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <vector>

#include "commux/process_group_ucx.hpp"

// c10::cuda::set_device lives in c10_cuda; its header pulls in a generated CUDA
// macro file that is absent from CPU-only torch, so guard it (and the call).
#ifdef COMMUX_WITH_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(
        stderr, "usage: %s <rank> <world_size> [host=127.0.0.1] [port=29555]\n",
        argv[0]);
    return 2;
  }
  int rank = std::atoi(argv[1]);
  int size = std::atoi(argv[2]);
  std::string host = argc > 3 ? argv[3] : "127.0.0.1";
  int port = argc > 4 ? std::atoi(argv[4]) : 29555;

  c10d::TCPStoreOptions sopts;
  sopts.port = static_cast<std::uint16_t>(port);
  sopts.isServer = (rank == 0);
  sopts.numWorkers = size;
  sopts.waitWorkers = true;
  auto store = c10::make_intrusive<c10d::TCPStore>(host, sopts);

  auto pg = c10::make_intrusive<commux::ProcessGroupUCX>(store, rank, size);

  const char* dev_env = std::getenv("UCXPG_DEVICE");
  bool want_cuda = dev_env != nullptr && std::string(dev_env) == "cuda";
  torch::Device device(torch::kCPU);
  if (want_cuda) {
    if (!torch::cuda::is_available()) {
      std::fprintf(stderr, "[rank %d] CUDA requested but not available\n",
                   rank);
      return 2;
    }
    int dev_index = rank % static_cast<int>(torch::cuda::device_count());
    device = torch::Device(torch::kCUDA, dev_index);
#ifdef COMMUX_WITH_CUDA
    c10::cuda::set_device(dev_index);
#endif
  }
  std::printf("[rank %d/%d] commux process group up (backend=%s, device=%s)\n",
              rank, size, pg->getBackendName().c_str(),
              want_cuda ? device.str().c_str() : "cpu");

  auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);

  // --- Test 1: tag-matched p2p between ranks 0 and 1 (non-blocking) --------
  const int TAG_A = 100;
  const int TAG_B = 200;
  if (size >= 2 && rank == 0) {
    auto a = torch::full({4}, 1.0, f64);
    auto b = torch::full({4}, 2.0, f64);
    std::vector<at::Tensor> va{a}, vb{b};
    auto wa = pg->send(va, /*dst=*/1, TAG_A);  // post A ...
    auto wb = pg->send(vb, /*dst=*/1, TAG_B);  // ... then B, both in flight
    wa->wait();
    wb->wait();
    std::printf("[rank 0] sent tag %d (=1.0) then tag %d (=2.0)\n", TAG_A,
                TAG_B);
  } else if (size >= 2 && rank == 1) {
    auto bufB = torch::zeros({4}, f64);
    auto bufA = torch::zeros({4}, f64);
    std::vector<at::Tensor> vB{bufB}, vA{bufA};
    auto wB = pg->recv(vB, /*src=*/0, TAG_B);  // request B first ...
    auto wA = pg->recv(vA, /*src=*/0, TAG_A);  // ... then A
    wB->wait();
    wA->wait();
    bool ok =
        bufB.eq(2.0).all().item<bool>() && bufA.eq(1.0).all().item<bool>();
    check(ok, "tag-matched p2p: out-of-order recv delivered by tag, not order");
  }

  pg->barrier()->wait();

  // --- Test 2: allreduce SUM / MIN / MAX ----------------------------------
  {
    double mine = static_cast<double>(rank + 1);
    double expect_sum = static_cast<double>(size) * (size + 1) / 2.0;
    auto run = [&](c10d::ReduceOp::RedOpType op) {
      auto t = torch::full({3}, mine, f64);
      std::vector<at::Tensor> v{t};
      c10d::AllreduceOptions o;
      o.reduceOp = op;
      pg->allreduce(v, o)->wait();
      return t[0].item<double>();
    };
    check(run(c10d::ReduceOp::SUM) == expect_sum, "allreduce SUM");
    check(run(c10d::ReduceOp::MIN) == 1.0, "allreduce MIN");
    check(run(c10d::ReduceOp::MAX) == static_cast<double>(size),
          "allreduce MAX");
  }

  // --- Test 3: reduce SUM to root 0 ---------------------------------------
  {
    auto t = torch::full({2}, static_cast<double>(rank + 1), f64);
    std::vector<at::Tensor> v{t};
    c10d::ReduceOptions o;
    o.reduceOp = c10d::ReduceOp::SUM;
    o.rootRank = 0;
    pg->reduce(v, o)->wait();
    if (rank == 0) {
      double expect = static_cast<double>(size) * (size + 1) / 2.0;
      check(t[0].item<double>() == expect, "reduce SUM lands on root");
    }
  }

  pg->barrier()->wait();

  // --- Test 4: coalesced multi-tensor vector p2p (IOV path) ----------------
  // One send()/recv() call carrying V=3 differently-shaped tensors. With
  // COMMUX_COALESCE unset/1 this drives the IOV single-message path; with
  // COMMUX_COALESCE=0 the per-tensor fallback. Both must deliver identical
  // data.
  if (size >= 2) {
    const int TAG_V = 300;
    const std::vector<int64_t> shapes{5, 1, 8};
    if (rank == 0) {
      std::vector<at::Tensor> v;
      for (size_t k = 0; k < shapes.size(); ++k)
        v.push_back(torch::full({shapes[k]}, static_cast<double>(10 + k), f64));
      pg->send(v, /*dst=*/1, TAG_V)->wait();
    } else if (rank == 1) {
      std::vector<at::Tensor> v;
      for (size_t k = 0; k < shapes.size(); ++k)
        v.push_back(torch::zeros({shapes[k]}, f64));
      pg->recv(v, /*src=*/0, TAG_V)->wait();
      bool ok = true;
      for (size_t k = 0; k < shapes.size(); ++k)
        ok = ok && v[k].eq(static_cast<double>(10 + k)).all().item<bool>();
      check(ok, "coalesced multi-tensor vector p2p delivers all V buffers");
    }
  }

  // --- Test 5: coalescing group window (COMMUX_GROUP=1) --------------------
  // Mirrors a halo step: defer a recv + a send inside startCoalescing/
  // endCoalescing, then wait the single aggregate Work.
  const char* group_env = std::getenv("COMMUX_GROUP");
  bool group_on = group_env && (std::string(group_env) == "1" ||
                                std::string(group_env) == "on");
  if (group_on && size >= 2 && (rank == 0 || rank == 1)) {
    const int TG_A = 410, TG_B = 420;
    int peer = rank == 0 ? 1 : 0;
    auto out = torch::zeros({4}, f64);
    auto mine = torch::full({4}, static_cast<double>(rank + 1), f64);
    std::vector<at::Tensor> rv{out}, sv{mine};
    pg->startCoalescing();
    if (rank == 0) {
      pg->recv(rv, peer, TG_B);
      pg->send(sv, peer, TG_A);
    } else {
      pg->recv(rv, peer, TG_A);
      pg->send(sv, peer, TG_B);
    }
    auto w = pg->endCoalescing();
    if (w) w->wait();
    check(out.eq(static_cast<double>(peer + 1)).all().item<bool>(),
          "coalescing group window: deferred recv/send completed");
  }

  pg->barrier()->wait();

  // --- Optional micro-benchmark (COMMUX_BENCH=1): round-trip latency of a
  // V-tensor vector ping-pong. Isolates message coalescing: the single stream-
  // sync applies to both paths, so COMMUX_COALESCE toggles only #messages
  // (1 vs V). COMMUX_BENCH_V sets V (default 16). -----------------------------
  if (std::getenv("COMMUX_BENCH") && size >= 2) {
    const int V = std::getenv("COMMUX_BENCH_V")
                      ? std::atoi(std::getenv("COMMUX_BENCH_V"))
                      : 16;
    const int ITERS = 2000;
    const int64_t N = 1024;
    std::vector<at::Tensor> sv, rv;
    for (int k = 0; k < V; ++k) {
      sv.push_back(torch::full({N}, static_cast<double>(k), f64));
      rv.push_back(torch::zeros({N}, f64));
    }
    const int TAG = 900;
    pg->barrier()->wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < ITERS; ++it) {
      if (rank == 0) {
        pg->send(sv, 1, TAG)->wait();
        pg->recv(rv, 1, TAG + 1)->wait();
      } else if (rank == 1) {
        pg->recv(rv, 0, TAG)->wait();
        pg->send(sv, 0, TAG + 1)->wait();
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    if (rank == 0) {
      double us =
          std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
      const char* c = std::getenv("COMMUX_COALESCE");
      std::printf(
          "[bench] V=%d iters=%d round-trip=%.2f us/iter "
          "(COMMUX_COALESCE=%s)\n",
          V, ITERS, us, c ? c : "default");
    }
    pg->barrier()->wait();
  }

  // --- Test 6: concurrent multi-threaded send/recv (COMMUX_MT=1) -----------
  // Two threads per rank drive the SAME process group concurrently, each
  // exchanging with the peer on its own tag for many iterations. Validates
  // thread-safe worker access (UCS_THREAD_MODE_MULTI) and the lock-free
  // blocking wait -- the old serialized, wait-holds-the-lock design would
  // serialize and could deadlock multi-threaded consumers such as snapy.
  if (std::getenv("COMMUX_MT") && size == 2) {
    const int NT = 2, ITERS = 200;
    int peer = 1 - rank;
    std::atomic<int> good{0};
    auto job = [&](int t) {
      bool ok = true;
      for (int it = 0; it < ITERS && ok; ++it) {
        double val = rank * 1000 + t * 10 + (it % 7);
        std::vector<at::Tensor> sv{torch::full({16}, val, f64)};
        std::vector<at::Tensor> rv{torch::zeros({16}, f64)};
        auto wr = pg->recv(rv, peer, 600 + t);
        auto ws = pg->send(sv, peer, 600 + t);
        wr->wait();
        ws->wait();
        ok = rv[0].eq(peer * 1000 + t * 10 + (it % 7)).all().item<bool>();
      }
      if (ok) good.fetch_add(1);
    };
    std::vector<std::thread> ths;
    for (int t = 0; t < NT; ++t) ths.emplace_back(job, t);
    for (auto& th : ths) th.join();
    check(good.load() == NT, "concurrent multi-threaded send/recv (2 threads)");
  }

  // --- Test 7: many threads parked in wait must all wake (no lost wakeup) ----
  // (COMMUX_DEADLOCK=1) Regression for the snapy hang. Several receiver threads
  // each post a recv and block at the SAME time, before the sender sends (it
  // delays), so all of them are genuinely parked in the wait concurrently. The
  // old wait slept in an infinite ucp_worker_wait(): when multiple threads are
  // armed on the one shared worker's wakeup fd, a single arriving event wakes
  // only ONE of them, leaving the others parked forever -- the lost-wakeup
  // deadlock that hit snapy's multi-threaded BlockWorkerPool driving the
  // worker. The periodic wait (worker_wait_timed) re-progresses on a short
  // timeout, so every thread makes progress regardless of who consumes the
  // event. A watchdog aborts on regression rather than hanging the suite.
  if (std::getenv("COMMUX_DEADLOCK") && size == 2) {
    const int NT = 8;  // threads parked on the shared worker at once
    const int64_t N = 1024;
    const int TG = 700;
    int peer = 1 - rank;
    std::atomic<bool> done{false};
    std::thread watchdog([&] {
      for (int i = 0; i < 600 && !done.load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      if (!done.load()) {
        std::fprintf(stderr,
                     "[rank %d] Test 7 TIMEOUT (30s): threads parked in wait "
                     "did not all wake (lost-wakeup regression)\n",
                     rank);
        std::abort();
      }
    });
    // Lost wakeups are a scheduling race, so repeat: any one losing round hangs
    // (and trips the watchdog). Each round spawns NT recv threads that all
    // park, then -- after a delay -- delivers every message in a single burst.
    // On the buggy wait one waking thread drains all NT completions via one
    // ucp_worker_progress() while the others stay asleep on the (already
    // consumed) wakeup fd, so they never re-check and hang.
    const int ROUNDS = 16;
    std::atomic<int> good{0};
    auto recv_job = [&](int t) {
      auto r = torch::zeros({N}, f64);
      std::vector<at::Tensor> v{r};
      pg->recv(v, peer, TG + t)->wait();  // parks; NT of these at once
      if (r.eq(static_cast<double>(peer * 100 + t)).all().item<bool>())
        good.fetch_add(1);
    };
    for (int round = 0; round < ROUNDS; ++round) {
      std::vector<std::thread> recvs;
      for (int t = 0; t < NT; ++t) recvs.emplace_back(recv_job, t);
      std::this_thread::sleep_for(std::chrono::milliseconds(400));  // let parks
      std::vector<at::Tensor> sbufs;
      std::vector<c10::intrusive_ptr<c10d::Work>> sw;
      for (int t = 0; t < NT; ++t) {  // burst: post all sends, then wait
        sbufs.push_back(
            torch::full({N}, static_cast<double>(rank * 100 + t), f64));
        std::vector<at::Tensor> v{sbufs.back()};
        sw.push_back(pg->send(v, peer, TG + t));
      }
      for (auto& w : sw) w->wait();
      for (auto& th : recvs) th.join();
    }
    check(good.load() == NT * ROUNDS,
          "concurrent parked waits all wake (no lost wakeup)");
    done.store(true);
    watchdog.join();
  }

  // COMMUX_SPIN_US only. A peer drips unrelated messages for longer than the
  // spin window, then sends the message this rank is blocked on. The whole
  // wait gets one window, so CPU time stays near that budget. Default-off
  // runs skip this.
  if (const char* spin_env = std::getenv("COMMUX_SPIN_US");
      spin_env != nullptr && std::atoi(spin_env) > 0 && size == 2) {
    const int peer = 1 - rank;
    const int n_noise = 40;
    const int gap_ms = 3;
    const int target_tag = 9100;
    std::vector<at::Tensor> noise_bufs;
    std::vector<c10::intrusive_ptr<c10d::Work>> noise_works;
    at::Tensor target;
    c10::intrusive_ptr<c10d::Work> target_work;
    if (rank == 0) {
      target = torch::zeros({8}, f64);
      std::vector<at::Tensor> tv{target};
      target_work = pg->recv(tv, peer, target_tag);
      for (int i = 0; i < n_noise; ++i) {
        noise_bufs.push_back(torch::zeros({8}, f64));
        std::vector<at::Tensor> v{noise_bufs.back()};
        noise_works.push_back(pg->recv(v, peer, 9200 + i));
      }
    }
    pg->barrier()->wait();
    if (rank == 0) {
      struct rusage ru0{};
      struct rusage ru1{};
      getrusage(RUSAGE_THREAD, &ru0);
      auto wall0 = std::chrono::steady_clock::now();
      target_work->wait();
      long wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - wall0)
                         .count();
      getrusage(RUSAGE_THREAD, &ru1);
      auto usec = [](const timeval& t) {
        return static_cast<long>(t.tv_sec) * 1000000L +
               static_cast<long>(t.tv_usec);
      };
      long cpu_us = (usec(ru1.ru_utime) - usec(ru0.ru_utime)) +
                    (usec(ru1.ru_stime) - usec(ru0.ru_stime));
      for (auto& w : noise_works) w->wait();
      bool value_ok = target.eq(42.0).all().item<bool>();
      long budget_us = std::atol(spin_env);
      long drip_us = static_cast<long>(n_noise) * gap_ms * 1000L;
      // A window longer than the drip spins until the message arrives.
      bool parked = budget_us * 2 >= drip_us || (wall_ms * 1000 > drip_us / 2 &&
                                                 cpu_us * 2 < wall_ms * 1000);
      std::printf("  spin check: wall %ld ms, cpu %ld us, budget %s us\n",
                  wall_ms, cpu_us, spin_env);
      check(value_ok && parked,
            "bounded spin parks while unrelated traffic flows");
    } else {
      for (int i = 0; i < n_noise; ++i) {
        auto s = torch::full({8}, 1.0, f64);
        std::vector<at::Tensor> v{s};
        pg->send(v, peer, 9200 + i)->wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
      }
      auto s = torch::full({8}, 42.0, f64);
      std::vector<at::Tensor> v{s};
      pg->send(v, peer, target_tag)->wait();
    }

    // Tight unrelated sends, no gap, held for longer than the budget. A short
    // burst finishes inside the budget and both libraries spin through it.
    // These recvs are posted only after the target arrives, so during the wait
    // the messages are unexpected and ucp_worker_progress stays non-zero. The
    // outer `if (progress) continue` has to read the deadline itself.
    const int flood_ms = 40;
    const int flood_max = 20000;
    const int flood_target = 9400;
    at::Tensor flood_got;
    c10::intrusive_ptr<c10d::Work> flood_work;
    if (rank == 0) {
      flood_got = torch::empty({8}, f64);
      std::vector<at::Tensor> tv{flood_got};
      flood_work = pg->recv(tv, peer, flood_target);
    }
    pg->barrier()->wait();
    if (rank == 0) {
      struct rusage ru0{};
      struct rusage ru1{};
      getrusage(RUSAGE_THREAD, &ru0);
      auto wall0 = std::chrono::steady_clock::now();
      flood_work->wait();
      long wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - wall0)
                         .count();
      getrusage(RUSAGE_THREAD, &ru1);
      auto usec = [](const timeval& t) {
        return static_cast<long>(t.tv_sec) * 1000000L +
               static_cast<long>(t.tv_usec);
      };
      long cpu_us = (usec(ru1.ru_utime) - usec(ru0.ru_utime)) +
                    (usec(ru1.ru_stime) - usec(ru0.ru_stime));
      int n = static_cast<int>(flood_got[0].item<double>());
      std::vector<at::Tensor> flood_bufs;
      std::vector<c10::intrusive_ptr<c10d::Work>> flood_works;
      if (n > 0) flood_bufs.reserve(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) {
        flood_bufs.push_back(torch::empty({8}, f64));
        std::vector<at::Tensor> v{flood_bufs.back()};
        flood_works.push_back(pg->recv(v, peer, 9500 + i));
      }
      for (auto& w : flood_works) w->wait();
      long budget_us = std::atol(spin_env);
      long wall_us = wall_ms * 1000L;
      // A flood that finishes inside the budget cannot fail a full spin.
      bool long_enough = wall_ms + 5 >= flood_ms;
      bool parked = long_enough && cpu_us * 2 < wall_us;
      std::printf(
          "  spin flood: wall %ld ms, cpu %ld us, budget %ld us, n %d\n",
          wall_ms, cpu_us, budget_us, n);
      check(
          n > 0 && parked,
          "bounded spin parks while unrelated completions arrive back to back");
    } else {
      int n = 0;
      auto end = std::chrono::steady_clock::now() +
                 std::chrono::milliseconds(flood_ms);
      while (std::chrono::steady_clock::now() < end && n < flood_max) {
        auto s = torch::full({8}, 1.0, f64);
        std::vector<at::Tensor> v{s};
        pg->send(v, peer, 9500 + n)->wait();
        ++n;
      }
      auto s = torch::full({8}, static_cast<double>(n), f64);
      std::vector<at::Tensor> v{s};
      pg->send(v, peer, flood_target)->wait();
    }
  }

  // COMMUX_LATE only. A wait that already outlasted COMMUX_SPIN_US must not
  // sit for another COMMUX_WAIT_POLL_MS after the request has completed. The
  // peer holds the message until this rank's budget is expired and the wait
  // is parked; the payload is the sender's steady_clock in microseconds,
  // which is the same CLOCK_MONOTONIC on one host. Red on 0994e4e: the
  // median is about one poll. COMMUX_WAIT_POLL_MS must be >= 10 so one poll
  // is larger than localhost noise. spin_us() and wait_poll_ms() cache the
  // environment on first use, so both variables have to be set at process
  // start, before any wait.
  if (const char* late = std::getenv("COMMUX_LATE");
      late != nullptr && std::atoi(late) > 0 && size == 2) {
    const char* poll_env = std::getenv("COMMUX_WAIT_POLL_MS");
    const char* spin_env = std::getenv("COMMUX_SPIN_US");
    int poll_ms = poll_env != nullptr ? std::atoi(poll_env) : 1;
    if (poll_ms <= 0) poll_ms = 1;
    long spin_us = spin_env != nullptr ? std::atol(spin_env) : 0;
    const int peer = 1 - rank;
    const int trials = 11;
    // Stay parked for longer than the budget before the message exists.
    const int hold_ms = static_cast<int>(spin_us / 1000L) + 8;
    auto steady_us = []() {
      return std::chrono::duration_cast<std::chrono::microseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
          .count();
    };
    if (rank == 0) {
      check(spin_us > 0, "COMMUX_LATE requires COMMUX_SPIN_US > 0");
      check(poll_ms >= 10, "COMMUX_LATE requires COMMUX_WAIT_POLL_MS >= 10");
      std::vector<long> lates;
      lates.reserve(static_cast<size_t>(trials));
      for (int i = 0; i < trials; ++i) {
        auto got = torch::empty({1}, f64);
        std::vector<at::Tensor> tv{got};
        auto work = pg->recv(tv, peer, 9700 + i);
        pg->barrier()->wait();
        work->wait();
        long done = steady_us();
        long sent = std::llround(got[0].item<double>());
        lates.push_back(done - sent);
      }
      std::sort(lates.begin(), lates.end());
      long median = lates[lates.size() / 2];
      long lo = lates.front();
      long hi = lates.back();
      std::printf(
          "  late poll: median %ld us, min %ld us, max %ld us, poll %d ms, "
          "hold %d ms, spin %ld us, n %d\n",
          median, lo, hi, poll_ms, hold_ms, spin_us, trials);
      // Half a poll leaves room for scheduling noise and still rejects a
      // wait that parked for a full COMMUX_WAIT_POLL_MS after completion.
      check(lo > 0 && median * 2 < static_cast<long>(poll_ms) * 1000L,
            "wait past the spin budget does not pay another poll");
    } else {
      for (int i = 0; i < trials; ++i) {
        pg->barrier()->wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
        auto sent = torch::full({1}, static_cast<double>(steady_us()), f64);
        std::vector<at::Tensor> v{sent};
        pg->send(v, peer, 9700 + i)->wait();
      }
    }
  }

  pg->barrier()->wait();
  if (rank == 0) {
    std::printf(
        g_failures == 0 ? "\nALL TESTS PASSED\n" : "\n%d TEST(S) FAILED\n",
        g_failures);
  }
  return g_failures == 0 ? 0 : 1;
}
