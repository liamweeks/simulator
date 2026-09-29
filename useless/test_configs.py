"""
Validation test matrix for the queue simulator.

Four models (M/M/1, M/M/c, M/D/1, M/D/c) at three load levels
(rho = 0.1, 0.5, 0.9), giving 12 configs.

Conventions
-----------
* Service rate mu = 1.0 everywhere, so mean service time is 1 time unit and
  every theoretical Wq below is expressed in service-time units.
* Single-server models: num_servers = 1, so lam = rho.
* Multi-server models: num_servers = 2 (c = 2), so lam = 2*rho.
* bulk_service is left at its default of 1 (no batch service) on every row;
  num_servers is the parallel-server count.
* rho is written as the exact expression lam / (mu * num_servers), in the same
  operand order the constructor uses, so the strict float-equality check
  (lam / (mu * num_servers) != rho) cannot trip on rounding.
* sim_time_limit and warm_up_period grow with rho because the relaxation time
  of these queues blows up as rho -> 1; a short run at rho = 0.9 biases the
  measured Wq low and leaves the tail under-sampled.

Theoretical mean Wq (service-time units), for checking recorder output:

              rho=0.1     rho=0.5     rho=0.9
  M/M/1       0.1111      1.0000      9.0000      Wq = rho / (1 - rho)
  M/D/1       0.0556      0.5000      4.5000      Wq = rho / (2*(1 - rho))   (P-K, half of M/M/1)
  M/M/2       0.0101      0.3333      4.2632      Wq = ErlangC / (c*mu - lam)
  M/D/2      ~0.0051     ~0.1667     ~2.1316      no closed form; ~0.5 * M/M/2
                                                  heuristic, exact via Crommelin
"""

import random

from queue_config import QueueConfig

CONFIGS = [
    # ------------------------------ M/M/1 ------------------------------
    QueueConfig(
        name="MM1_rho0.1",
        lam=0.1,
        mu=1.0,
        num_servers=1,
        arrival_distribution=lambda: random.expovariate(0.1),
        service_distribution=lambda: random.expovariate(1.0),
        rho=0.1 / (1.0 * 1),
        sim_time_limit=5e4,
        warm_up_period=2e3,
        bulk_service=1,
    ),  # theoretical Wq = 0.1111
    QueueConfig(
        name="MM1_rho0.5",
        lam=0.5,
        mu=1.0,
        num_servers=1,
        arrival_distribution=lambda: random.expovariate(0.5),
        service_distribution=lambda: random.expovariate(1.0),
        rho=0.5 / (1.0 * 1),
        sim_time_limit=5e4,
        warm_up_period=5e3,
        bulk_service=1,
    ),  # theoretical Wq = 1.0000
    QueueConfig(
        name="MM1_rho0.9",
        lam=0.9,
        mu=1.0,
        num_servers=1,
        arrival_distribution=lambda: random.expovariate(0.9),
        service_distribution=lambda: random.expovariate(1.0),
        rho=0.9 / (1.0 * 1),
        sim_time_limit=2e5,
        warm_up_period=4e4,
        bulk_service=1,
    ),  # theoretical Wq = 9.0000

    # ------------------------------ M/D/1 ------------------------------
    QueueConfig(
        name="MD1_rho0.1",
        lam=0.1,
        mu=1.0,
        num_servers=1,
        arrival_distribution=lambda: random.expovariate(0.1),
        service_distribution=lambda: 1.0 / 1.0,
        rho=0.1 / (1.0 * 1),
        sim_time_limit=5e4,
        warm_up_period=2e3,
        bulk_service=1,
    ),  # theoretical Wq = 0.0556
    QueueConfig(
        name="MD1_rho0.5",
        lam=0.5,
        mu=1.0,
        num_servers=1,
        arrival_distribution=lambda: random.expovariate(0.5),
        service_distribution=lambda: 1.0 / 1.0,
        rho=0.5 / (1.0 * 1),
        sim_time_limit=5e4,
        warm_up_period=5e3,
        bulk_service=1,
    ),  # theoretical Wq = 0.5000
    QueueConfig(
        name="MD1_rho0.9",
        lam=0.9,
        mu=1.0,
        num_servers=1,
        arrival_distribution=lambda: random.expovariate(0.9),
        service_distribution=lambda: 1.0 / 1.0,
        rho=0.9 / (1.0 * 1),
        sim_time_limit=2e5,
        warm_up_period=4e4,
        bulk_service=1,
    ),  # theoretical Wq = 4.5000

    # ------------------------ M/M/c  (c = 2) ---------------------------
    QueueConfig(
        name="MM2_rho0.1",
        lam=0.2,
        mu=1.0,
        num_servers=2,
        arrival_distribution=lambda: random.expovariate(0.2),
        service_distribution=lambda: random.expovariate(1.0),
        rho=0.2 / (1.0 * 2),
        sim_time_limit=5e4,
        warm_up_period=2e3,
        bulk_service=1,
    ),  # theoretical Wq = 0.0101
    QueueConfig(
        name="MM2_rho0.5",
        lam=1.0,
        mu=1.0,
        num_servers=2,
        arrival_distribution=lambda: random.expovariate(1.0),
        service_distribution=lambda: random.expovariate(1.0),
        rho=1.0 / (1.0 * 2),
        sim_time_limit=5e4,
        warm_up_period=5e3,
        bulk_service=1,
    ),  # theoretical Wq = 0.3333
    QueueConfig(
        name="MM2_rho0.9",
        lam=1.8,
        mu=1.0,
        num_servers=2,
        arrival_distribution=lambda: random.expovariate(1.8),
        service_distribution=lambda: random.expovariate(1.0),
        rho=1.8 / (1.0 * 2),
        sim_time_limit=2e5,
        warm_up_period=4e4,
        bulk_service=1,
    ),  # theoretical Wq = 4.2632

    # ------------------------ M/D/c  (c = 2) ---------------------------
    QueueConfig(
        name="MD2_rho0.1",
        lam=0.2,
        mu=1.0,
        num_servers=2,
        arrival_distribution=lambda: random.expovariate(0.2),
        service_distribution=lambda: 1.0 / 1.0,
        rho=0.2 / (1.0 * 2),
        sim_time_limit=5e4,
        warm_up_period=2e3,
        bulk_service=1,
    ),  # theoretical Wq ~ 0.0051 (approx)
    QueueConfig(
        name="MD2_rho0.5",
        lam=1.0,
        mu=1.0,
        num_servers=2,
        arrival_distribution=lambda: random.expovariate(1.0),
        service_distribution=lambda: 1.0 / 1.0,
        rho=1.0 / (1.0 * 2),
        sim_time_limit=5e4,
        warm_up_period=5e3,
        bulk_service=1,
    ),  # theoretical Wq ~ 0.1667 (approx)
    QueueConfig(
        name="MD2_rho0.9",
        lam=1.8,
        mu=1.0,
        num_servers=2,
        arrival_distribution=lambda: random.expovariate(1.8),
        service_distribution=lambda: 1.0 / 1.0,
        rho=1.8 / (1.0 * 2),
        sim_time_limit=2e5,
        warm_up_period=4e4,
        bulk_service=1,
    ),  # theoretical Wq ~ 2.1316 (approx)
]