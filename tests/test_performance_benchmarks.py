"""Performance benchmarks validating memory stability, no_grad coverage, and latency targets."""
import time
import tracemalloc
import pytest
import torch
from unittest.mock import patch

from src.config import LRSGConfig, LocomotionConfig, EmbodiedConfig
from src.simulation.embodied_loop import EmbodiedAgent


SMALL_CONFIG = dict(
    lrsg_config=LRSGConfig(num_neurons=5000),
    locomotion_config=LocomotionConfig(),
    embodied_config=EmbodiedConfig(grid_size=10, num_obstacles=1, episode_length=20),
    device='cpu',
)

LATENCY_LIMIT_MS = 100.0
MEMORY_GROWTH_LIMIT_KB = 2048  # allow up to 2MB growth over 5 episodes


def make_agent():
    return EmbodiedAgent(**SMALL_CONFIG)


class TestNoGradCoverage:
    """Verify torch.no_grad() wraps every forward_pass call during inference."""

    def test_forward_pass_called_without_grad(self):
        """forward_pass must be called with grad disabled in EmbodiedAgent.step()."""
        agent = make_agent()
        grad_enabled_during_call = []

        original_fp = __import__('src.lrsg.forward', fromlist=['forward_pass']).forward_pass

        def spy_forward_pass(*args, **kwargs):
            grad_enabled_during_call.append(torch.is_grad_enabled())
            return original_fp(*args, **kwargs)

        with patch('src.simulation.embodied_loop.forward_pass', spy_forward_pass):
            agent.step()

        assert len(grad_enabled_during_call) >= 1, "forward_pass was never called"
        assert not any(grad_enabled_during_call), (
            f"forward_pass ran WITH grad enabled on {sum(grad_enabled_during_call)} call(s). "
            "Every inference call must be inside torch.no_grad()."
        )

    def test_forward_pass_no_grad_across_10_steps(self):
        """Verify no_grad holds across 10 consecutive steps (multi-tick)."""
        agent = make_agent()
        violations = []

        original_fp = __import__('src.lrsg.forward', fromlist=['forward_pass']).forward_pass

        def spy_forward_pass(*args, **kwargs):
            if torch.is_grad_enabled():
                violations.append(True)
            return original_fp(*args, **kwargs)

        with patch('src.simulation.embodied_loop.forward_pass', spy_forward_pass):
            for _ in range(10):
                agent.step()

        assert not violations, (
            f"torch.no_grad() was missing on {len(violations)} forward_pass call(s) out of 10 steps."
        )

    def test_no_autograd_tensors_in_lrsg_after_step(self):
        """LRSG activations must not accumulate grad_fn after stepping."""
        agent = make_agent()
        agent.step()

        # No activation tensor should carry a computation graph
        graph = agent.lrsg
        assert graph.activations.grad_fn is None, (
            "activations has grad_fn — autograd graph is being retained across steps"
        )


class TestLatency:
    """Per-step latency must stay within the 100ms target."""

    def test_single_step_latency_under_100ms(self):
        """A single step should complete in under 100ms on CPU."""
        agent = make_agent()
        # Warm-up to avoid initialization costs in measurement
        agent.step()

        start = time.perf_counter()
        agent.step()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < LATENCY_LIMIT_MS, (
            f"Single step took {elapsed_ms:.1f}ms, exceeds {LATENCY_LIMIT_MS}ms target"
        )

    def test_median_step_latency_under_100ms(self):
        """Median latency over 20 steps must be under 100ms."""
        agent = make_agent()
        agent.step()  # warm-up

        latencies_ms = []
        for _ in range(20):
            t0 = time.perf_counter()
            agent.step()
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        median_ms = sorted(latencies_ms)[len(latencies_ms) // 2]
        max_ms = max(latencies_ms)

        # Attach timing summary to pytest output
        print(
            f"\n[perf] Step latency over 20 steps: "
            f"median={median_ms:.1f}ms  max={max_ms:.1f}ms  "
            f"p95={sorted(latencies_ms)[int(0.95 * len(latencies_ms))]:.1f}ms"
        )

        assert median_ms < LATENCY_LIMIT_MS, (
            f"Median step latency {median_ms:.1f}ms exceeds {LATENCY_LIMIT_MS}ms target"
        )


class TestMemoryStability:
    """Memory usage must remain bounded across multiple episodes."""

    def test_cpu_memory_bounded_over_5_episodes(self):
        """RSS memory growth must stay under 2MB over 5 complete episodes."""
        import psutil, os

        agent = make_agent()
        proc = psutil.Process(os.getpid())

        # Run one warm-up episode
        for _ in range(SMALL_CONFIG['embodied_config'].episode_length):
            agent.step()

        mem_before_kb = proc.memory_info().rss / 1024

        # Run 5 full episodes
        steps_per_episode = SMALL_CONFIG['embodied_config'].episode_length
        for _ in range(5 * steps_per_episode):
            agent.step()

        mem_after_kb = proc.memory_info().rss / 1024
        growth_kb = mem_after_kb - mem_before_kb

        print(
            f"\n[perf] Memory: before={mem_before_kb:.0f}KB  "
            f"after={mem_after_kb:.0f}KB  growth={growth_kb:.0f}KB"
        )

        assert growth_kb < MEMORY_GROWTH_LIMIT_KB, (
            f"Memory grew by {growth_kb:.0f}KB over 5 episodes, "
            f"exceeds {MEMORY_GROWTH_LIMIT_KB}KB limit"
        )

    def test_tracemalloc_no_leak_over_episodes(self):
        """Python allocations must not grow unboundedly across 5 episodes."""
        agent = make_agent()
        steps_per_episode = SMALL_CONFIG['embodied_config'].episode_length

        # Warm-up
        for _ in range(steps_per_episode):
            agent.step()

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for _ in range(5 * steps_per_episode):
            agent.step()

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_new_kb = sum(s.size_diff for s in top_stats if s.size_diff > 0) / 1024

        print(f"\n[perf] tracemalloc new allocations over 5 episodes: {total_new_kb:.0f}KB")

        assert total_new_kb < MEMORY_GROWTH_LIMIT_KB, (
            f"Python heap grew by {total_new_kb:.0f}KB over 5 episodes, "
            f"exceeds {MEMORY_GROWTH_LIMIT_KB}KB limit"
        )


class TestGPUPerformance:
    """GPU-specific performance benchmarks (skipped when CUDA is unavailable)."""

    @pytest.fixture(autouse=True)
    def require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_gpu_memory_delta_per_episode_near_zero(self):
        """GPU memory should not grow between episodes (no graph retention)."""
        agent = EmbodiedAgent(
            lrsg_config=LRSGConfig(num_neurons=5000),
            locomotion_config=LocomotionConfig(),
            embodied_config=EmbodiedConfig(grid_size=10, num_obstacles=1, episode_length=20),
            device='cuda',
        )
        steps = SMALL_CONFIG['embodied_config'].episode_length

        # Warm-up episode
        for _ in range(steps):
            agent.step()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # One more episode
        for _ in range(steps):
            agent.step()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()

        delta_kb = (mem_after - mem_before) / 1024
        print(f"\n[perf] GPU memory delta per episode: {delta_kb:.1f}KB")

        # Allow small housekeeping allocations (e.g., RNG state), but not graph growth
        assert delta_kb < 512, (
            f"GPU memory grew by {delta_kb:.1f}KB per episode — autograd graph may be leaking"
        )

    def test_gpu_utilization_under_85_percent(self):
        """GPU utilization must not exceed 85% during a run."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pytest.skip("pynvml not available or no NVIDIA GPU")

        agent = EmbodiedAgent(
            lrsg_config=LRSGConfig(num_neurons=5000),
            locomotion_config=LocomotionConfig(),
            embodied_config=EmbodiedConfig(grid_size=10, num_obstacles=1, episode_length=20),
            device='cuda',
        )

        util_samples = []
        for _ in range(20):
            agent.step()
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            util_samples.append(util)

        pynvml.nvmlShutdown()

        peak_util = max(util_samples)
        avg_util = sum(util_samples) / len(util_samples)
        print(f"\n[perf] GPU utilization: avg={avg_util:.1f}%  peak={peak_util:.1f}%")

        assert peak_util <= 85, (
            f"GPU utilization peaked at {peak_util}%, exceeds 85% target"
        )
