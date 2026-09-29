"""
SimPy simulator for the M/D^(a,b)/1/N general bulk service queue
(Chaudhry & Gai, INFOR 50(2), 2012).

Model
-----
- Arrivals: Poisson with rate `lam`.
- Server: single server, general bulk service rule (a, b).
    * If fewer than `a` customers are waiting when the server becomes free,
      it idles until the queue reaches `a` (the quorum).
    * It then takes min(queue length, b) customers into service together.
    * Customers arriving during a service wait for the next batch.
- Service: deterministic. By default the service time depends on the batch
  size j (the "D_j" model): d_j = j / mu. Pass `service_time=lambda j: d`
  for a constant service time regardless of batch size.
- Capacity: `N` is the waiting-room size (the paper's B). Customers in
  service do not count against it, so the total system capacity is N + b.
  An arrival that finds N customers waiting is lost. Use N=math.inf for
  the infinite-buffer model.

Output
------
The post-departure queue-length distribution pi_n (the number waiting
immediately after each batch departs), which is what the paper's tables
and figures report, plus the blocking probability and mean waiting time.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import simpy


@dataclass
class BulkQueueConfig:
    lam: float                      # arrival rate
    a: int                          # quorum (minimum batch size)
    b: int                          # maximum batch size
    N: float = math.inf             # waiting-room capacity (paper's B)
    mu: float = 3.0                 # per-customer service rate, used by default d_j = j/mu
    service_time: Optional[Callable[[int], float]] = None  # d_j as a function of batch size j
    sim_time: float = 5e5
    warm_up: float = 5e3
    seed: Optional[int] = None
    name: str = ""

    def __post_init__(self):
        if not (1 <= self.a <= self.b):
            raise ValueError("Need 1 <= a <= b")
        if self.service_time is None:
            mu = self.mu
            self.service_time = lambda j: j / mu

    @property
    def rho(self) -> float:
        """Traffic intensity rho = lam * d_b / b (equals lam/mu when d_j = j/mu)."""
        return self.lam * self.service_time(self.b) / self.b


@dataclass
class SimResult:
    config: BulkQueueConfig
    departure_states: np.ndarray        # queue length after each post-warm-up departure
    batch_sizes: np.ndarray
    waits: np.ndarray                   # time from arrival to service start
    arrivals: int
    lost: int

    def pi(self, n_max: Optional[int] = None) -> np.ndarray:
        """Post-departure queue-length distribution pi_0 ... pi_{n_max}."""
        if n_max is None:
            n_max = int(self.config.N) if math.isfinite(self.config.N) else int(self.departure_states.max())
        counts = np.bincount(self.departure_states, minlength=n_max + 1)[: n_max + 1]
        return counts / len(self.departure_states)

    @property
    def mean(self) -> float:
        return float(self.departure_states.mean())

    @property
    def variance(self) -> float:
        return float(self.departure_states.var())

    @property
    def blocking_probability(self) -> float:
        return self.lost / self.arrivals if self.arrivals else 0.0

    def summary(self) -> str:
        c = self.config
        return (
            f"{c.name or 'M/D^(a,b)/1/N'}  (lam={c.lam}, a={c.a}, b={c.b}, N={c.N}, rho={c.rho:.4f})\n"
            f"  departures recorded : {len(self.departure_states):,}\n"
            f"  mean queue (post-dep): {self.mean:.6f}\n"
            f"  variance            : {self.variance:.6f}\n"
            f"  mean batch size     : {self.batch_sizes.mean():.4f}\n"
            f"  mean wait           : {self.waits.mean():.4f}\n"
            f"  blocking probability: {self.blocking_probability:.6g}"
        )


class BulkQueue:
    def __init__(self, config: BulkQueueConfig):
        self.cfg = config
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(config.seed)

        self.waiting: deque[float] = deque()     # arrival times of waiting customers
        self._quorum_reached: Optional[simpy.Event] = None

        self.departure_states: list[int] = []
        self.batch_sizes: list[int] = []
        self.waits: list[float] = []
        self.arrivals = 0
        self.lost = 0

    # -- processes ---------------------------------------------------------

    def arrival_process(self):
        cfg = self.cfg
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / cfg.lam))
            now = self.env.now
            counting = now >= cfg.warm_up
            if counting:
                self.arrivals += 1

            if len(self.waiting) >= cfg.N:
                if counting:
                    self.lost += 1
                continue

            self.waiting.append(now)

            # Wake an idle server once the quorum is met.
            if self._quorum_reached is not None and len(self.waiting) >= cfg.a:
                self._quorum_reached.succeed()
                self._quorum_reached = None

    def server_process(self):
        cfg = self.cfg
        while True:
            if len(self.waiting) < cfg.a:
                self._quorum_reached = self.env.event()
                yield self._quorum_reached

            j = min(len(self.waiting), cfg.b)
            start = self.env.now
            batch = [self.waiting.popleft() for _ in range(j)]

            yield self.env.timeout(cfg.service_time(j))

            if self.env.now >= cfg.warm_up:
                self.departure_states.append(len(self.waiting))
                self.batch_sizes.append(j)
                self.waits.extend(start - t for t in batch)

    # -- driver ------------------------------------------------------------

    def run(self) -> SimResult:
        self.env.process(self.arrival_process())
        self.env.process(self.server_process())
        self.env.run(until=self.cfg.sim_time)
        return SimResult(
            config=self.cfg,
            departure_states=np.asarray(self.departure_states, dtype=int),
            batch_sizes=np.asarray(self.batch_sizes, dtype=int),
            waits=np.asarray(self.waits, dtype=float),
            arrivals=self.arrivals,
            lost=self.lost,
        )


def simulate(config: BulkQueueConfig) -> SimResult:
    return BulkQueue(config).run()


def replicate(config: BulkQueueConfig, n_reps: int = 10, base_seed: int = 0):
    """Independent replications; returns (mean pi, 95% CI half-width per state, results)."""
    results = []
    for r in range(n_reps):
        cfg = BulkQueueConfig(**{**config.__dict__, "seed": base_seed + r})
        results.append(simulate(cfg))
    n_max = max(len(res.pi()) for res in results) - 1
    pis = np.vstack([res.pi(n_max) for res in results])
    half_width = 1.96 * pis.std(axis=0, ddof=1) / math.sqrt(n_reps)
    return pis.mean(axis=0), half_width, results


# ---------------------------------------------------------------------------
# Example: reproduce Table 3 and Figure 3 of Chaudhry & Gai (2012)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    TABLE3 = {  # rho: (lam, {n: pi_n}, mean)
        1 / 6: (0.5, {0: 0.035674, 1: 0.118913, 10: 0.001665, 20: 4.25e-10}, 3.333333),
        0.6:   (1.8, {1: 0.000073, 10: 0.104566, 20: 0.009811}, 12.014419),
        0.9:   (2.7, {1: 0.010436, 10: 0.075931, 20: 0.011430}, 19.971430),
    }

    print("Table 3: M/D_j^(20,40)/1, B = infinity, d_j = j/3\n")
    for rho, (lam, ref, ref_mean) in TABLE3.items():
        t0 = time.perf_counter()
        res = simulate(BulkQueueConfig(lam=lam, a=20, b=40, N=math.inf, mu=3, seed=1,
                                       name=f"rho={rho:.4g}"))
        pi = res.pi(60)
        print(res.summary())
        print(f"  paper mean          : {ref_mean:.6f}")
        for n, p in ref.items():
            print(f"    pi_{n:<3} sim={pi[n]:.6f}  paper={p:.6g}")
        print(f"  ({time.perf_counter() - t0:.1f}s)\n")

    # Figure 3: finite buffer B = 60
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lam, label, marker in [(0.5, "ρ=1/6", "o"), (1.5, "ρ=0.5", "s"), (2.7, "ρ=0.9", "*")]:
            res = simulate(BulkQueueConfig(lam=lam, a=20, b=40, N=60, mu=3, seed=2))
            ax.plot(res.pi(60), marker=marker, markersize=4, linewidth=1, label=label)
        ax.set_xlabel("Queue length")
        ax.set_ylabel("Probability")
        ax.set_xlim(0, 60)
        ax.set_title("Post-departure queue length, M/D_j^(20,40)/1/(60+40)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figure3_reproduction.png", dpi=200)
        print("Saved figure3_reproduction.png")
    except ImportError:
        pass