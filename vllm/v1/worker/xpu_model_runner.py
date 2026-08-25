# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import partial

import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.utils.torch_utils import supports_xpu_graph
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import (
    GPUModelRunner as GPUModelRunnerV2,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _env_on(name: str, default: str = "OFF") -> bool:
    return os.environ.get(name, default).upper() in ("1", "Y", "ON", "YES", "TRUE")


class ProfileTraceMixin:
    """Step timing (host) + torch.profiler kernel capture for XPU runners.

    Env vars
    --------
    TRACE=1            enable per-step host timing logs
    TRACE_SYNC=0|1     1 (default): sync at phase boundaries -> wall-clock per
                       phase (includes GPU wait). 0: no sync -> pure host/CPU
                       launch cost. Use 0 when hunting host bottlenecks.
    PROFILE=1          enable kernel capture around the model forward
    PROFILE_BACKEND    'torch' (default) uses torch.profiler and writes a
                       chrome trace per capture. 'unitrace' instead gates
                       PTI_ENABLE_COLLECTION so an external unitrace run
                       only collects these windows.
    PROFILE_PATH=dir   output dir for traces (default ./logs/)
    PROFILE_INTERVAL=n capture every n decode steps (default 200)
    NUM_WARMUPS=n      steps to skip before counting/capturing (default 0)
    PROFILE_ALL_RANKS=1 capture on every TP rank (default: rank0 only)
    PROFILE_STACK=1    record python stacks (13x bigger trace, 3x slower)
    """

    def _init_prof_state(self) -> None:
        self.is_trace = _env_on("TRACE")
        self.trace_sync = _env_on("TRACE_SYNC", "1")
        self.profile_env = _env_on("PROFILE")
        self.profile_path = os.environ.get("PROFILE_PATH", "./logs/")
        self.profile_every_n_steps = int(os.environ.get("PROFILE_INTERVAL", 200))
        self.warm_ups_num = int(os.environ.get("NUM_WARMUPS", 0))
        self.profile_all_ranks = _env_on("PROFILE_ALL_RANKS")
        # Python stacks make traces ~13x bigger and dominate export cost
        # (measured: 80MB/14.5s vs 6MB/5.1s) with identical kernel timings.
        # Only enable to attribute kernels back to source lines.
        self.profile_stack = _env_on("PROFILE_STACK")
        self.profile_backend = os.environ.get("PROFILE_BACKEND", "torch").lower()
        if self.profile_backend not in ("torch", "unitrace"):
            raise ValueError(
                f"PROFILE_BACKEND must be 'torch' or 'unitrace', "
                f"got {self.profile_backend!r}"
            )
        self.step = -1
        self.has_skip_warmup = self.warm_ups_num <= 0
        self._prof_t0 = 0.0
        self._prof_tprev = 0.0
        self._prof_do_trace = False
        self._prof_do_profile = False
        self._prof_m = 0
        self._prof_ev_pending: deque = deque()
        if self.is_trace or self.profile_env:
            os.makedirs(self.profile_path, exist_ok=True)

    # -- helpers ---------------------------------------------------------
    def _prof_is_rank0(self) -> bool:
        try:
            return get_tp_group().rank == 0
        except Exception:
            return True

    def _prof_now(self) -> float:
        if self.trace_sync:
            torch.xpu.synchronize()
        return time.perf_counter()

    def _log_schedule(self, scheduler_output: "SchedulerOutput") -> None:
        num_scheduled = scheduler_output.num_scheduled_tokens
        logger.info("total_requests_num: %d", len(num_scheduled))
        logger.info(
            "total_num_scheduled_tokens: %d", scheduler_output.total_num_scheduled_tokens
        )
        for req in scheduler_output.scheduled_new_reqs:
            logger.info("    request id: %s", req.req_id)
            logger.info("        context_len: %s", req.num_computed_tokens)
            logger.info("        num_scheduled_token: %s", num_scheduled[req.req_id])
        cached = scheduler_output.scheduled_cached_reqs
        for i, rid in enumerate(cached.req_ids):
            logger.info("    request id: %s", rid)
            logger.info("        context_len: %s", cached.num_computed_tokens[i])
            logger.info("        num_scheduled_token: %s", num_scheduled[rid])

    # -- hooks called from GPUModelRunner.execute_model -------------------
    def prof_begin_step(self, scheduler_output: "SchedulerOutput") -> None:
        if not (self.is_trace or self.profile_env):
            self._prof_do_trace = self._prof_do_profile = False
            return

        self.step += 1
        if not self.has_skip_warmup and self.step >= self.warm_ups_num:
            self.has_skip_warmup = True
            self.step = 0
            logger.info(
                "Finished %d warmup steps; tracing/profiling starts now.",
                self.warm_ups_num,
            )

        self._prof_m = scheduler_output.total_num_scheduled_tokens
        self._prof_do_trace = self.is_trace and self.has_skip_warmup

        has_prefill = any(
            scheduler_output.num_scheduled_tokens[r.req_id] > 1
            for r in scheduler_output.scheduled_new_reqs
        )
        rank_ok = self.profile_all_ranks or self._prof_is_rank0()
        self._prof_do_profile = (
            self.profile_env
            and self.has_skip_warmup
            and rank_ok
            and (has_prefill or self.step % self.profile_every_n_steps == 0)
        )

        if self._prof_do_trace and self._prof_is_rank0():
            logger.info("m = %d, step = %d:", self._prof_m, self.step)
            self._log_schedule(scheduler_output)

        if self._prof_do_trace:
            self._drain_device_events()
            self._prof_t0 = self._prof_now()
            self._prof_tprev = self._prof_t0

    def prof_mark(self, tag: str) -> None:
        if not self._prof_do_trace:
            return
        now = self._prof_now()
        if self._prof_is_rank0():
            logger.info("%s time: %.3f ms", tag, (now - self._prof_tprev) * 1000)
            if tag == "postprocess":
                logger.info(
                    "execute_model time: %.3f ms", (now - self._prof_t0) * 1000
                )
        self._prof_tprev = now

    def _drain_device_events(self) -> None:
        """Report finished forward device timings without blocking the host.

        Events are read only once ``query()`` says the work is done, so this
        never stalls the pipeline the way a synchronize() would. The queue is
        capped so a stuck event cannot grow it without bound.
        """
        q = self._prof_ev_pending
        while q and q[0][2].query():
            step, m, _, start, end = q.popleft()
            if self._prof_is_rank0():
                logger.info(
                    "step = %d, m = %d, forward device: %.3f ms",
                    step,
                    m,
                    start.elapsed_time(end),
                )
        if len(q) > 32:
            # Backlog: force the oldest out rather than leak events.
            step, m, end, start, _ = q.popleft()
            end.synchronize()
            if self._prof_is_rank0():
                logger.info(
                    "step = %d, m = %d, forward device: %.3f ms",
                    step,
                    m,
                    start.elapsed_time(end),
                )

    def prof_sample_tokens(self, super_call):
        """Wrap sample_tokens() so the step total covers sampling too.

        execute_model() only covers roughly 2/3 of the worker-side step cost;
        sampling is a separate call made after it returns.
        """
        if not self._prof_do_trace:
            return super_call()
        t0 = self._prof_now()
        try:
            return super_call()
        finally:
            now = self._prof_now()
            if self._prof_is_rank0():
                logger.info("sample_tokens time: %.3f ms", (now - t0) * 1000)
                logger.info(
                    "step time (worker): %.3f ms", (now - self._prof_t0) * 1000
                )

    @contextmanager
    def prof_forward_ctx(self):
        """Wrap the model forward with XPU events (device time) and,
        on checkpoint steps, the torch profiler.

        Device time is what a projection model should be compared against;
        the host wall-clock printed by prof_mark() is a different quantity
        (on small/eager models it can be ~6x larger).
        """
        start = end = None
        if self._prof_do_trace:
            start = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            start.record()
        try:
            if self._prof_do_profile:
                capture = (
                    self._unitrace_ctx()
                    if self.profile_backend == "unitrace"
                    else self._torch_profiler_ctx()
                )
                with capture:
                    yield
            else:
                yield
        finally:
            if self._prof_do_trace:
                end.record()
                self._prof_ev_pending.append(
                    (self.step, self._prof_m, end, start, end)
                )

    @contextmanager
    def _torch_profiler_ctx(self):
        from torch.profiler import ProfilerActivity, profile

        os.environ["PRINT_EXPERTS"] = "1"
        torch.xpu.synchronize()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            with_stack=self.profile_stack,
            with_modules=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        ) as prof:
            yield
            torch.xpu.synchronize()
        os.environ["PRINT_EXPERTS"] = "0"

        rank = get_tp_group().rank
        base = f"rank{rank}_step_{self.step}_m_{self._prof_m}"
        with open(f"{self.profile_path}/vllm_profile_{base}.txt", "w") as f:
            f.write(prof.key_averages().table(sort_by="self_xpu_time_total"))
        prof.export_chrome_trace(f"{self.profile_path}/vllm_trace_{base}.json")
        logger.info("Saved torch profiler trace: vllm_trace_%s.json", base)


    @contextmanager
    def _unitrace_ctx(self):
        """Gate PTI collection to just this forward.

        Expects the process to run under an external collector, e.g.
        `unitrace --chrome-kernel-logging ... python -m vllm...`, started with
        PTI_ENABLE_COLLECTION=0 so only these windows are recorded.
        """
        os.environ["PRINT_EXPERTS"] = "1"
        torch.xpu.synchronize()
        os.environ["PTI_ENABLE_COLLECTION"] = "1"
        try:
            yield
        finally:
            torch.xpu.synchronize()
            os.environ["PTI_ENABLE_COLLECTION"] = "0"
            os.environ["PRINT_EXPERTS"] = "0"
            logger.info(
                "unitrace collection window: rank%d step %d m %d",
                get_tp_group().rank,
                self.step,
                self._prof_m,
            )


class XPUModelRunner(ProfileTraceMixin, GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        self._init_prof_state()
        # FIXME: To be verified.
        self.cascade_attn_enabled = False

    def sample_tokens(self, grammar_output):
        return self.prof_sample_tokens(
            lambda: super(XPUModelRunner, self).sample_tokens(grammar_output)
        )


class XPUModelRunnerV2(GPUModelRunnerV2):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)


@contextmanager
def _torch_cuda_wrapper():
    # Replace cuda APIs with xpu APIs. Each callable gets its own functools.partial
    # so it is not the same object as torch.xpu.* (Torch Dynamo _get_handlers()
    # asserts on duplicate registration when cuda aliases xpu directly).
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.default_stream = partial(torch.xpu.current_stream)
    torch.cuda.current_stream = partial(torch.xpu.current_stream)
    torch.cuda.stream = partial(torch.xpu.stream)
    torch.cuda.set_stream = partial(torch.xpu.set_stream)

    # torch.xpu.Event does not accept the ``blocking`` kwarg that
    # torch.cuda.Event supports, so drop it here.
    def _xpu_event(*args, blocking=None, **kwargs):
        return torch.xpu.Event(*args, **kwargs)

    torch.cuda.Event = _xpu_event
    if supports_xpu_graph():
        torch.cuda.graph = partial(torch.xpu.graph)
        torch.cuda.CUDAGraph = torch.xpu.XPUGraph
        torch.cuda.graph_pool_handle = partial(torch.xpu.graph_pool_handle)
        torch.cuda.is_current_stream_capturing = torch.xpu.is_current_stream_capturing
    yield
