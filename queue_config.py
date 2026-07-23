import math
from typing import Optional, Callable
import sys

class QueueConfig:
    """
    The QueueConfig class bundles the parameters for each queueing model.
    If the service_distribution is None, the model will default to an Erlang
    distribution to model the service-time distribution.
    """
    def __init__(self,
                 name="queue_model",
                 sim_time_limit=math.inf,
                 packet_limit=math.inf,
                 warm_up_period=5000,
                 num_servers=1,
                 arrival_distribution=None,
                 service_distribution=None,
                 lam=None,
                 mu=None,
                 rho=None,
                 max_queue_length=math.inf,
                 quorum: Optional[int] = None,  # a
                 bulk_service: Optional[int]=None, # b

                 ):
        self.packet_limit: Optional[int] = packet_limit
        self.lam: Optional[float] = lam
        self.mu: Optional[float] = mu
        self.rho: Optional[float] = rho
        self.arrival_distribution: Optional[Callable[[], float]] = arrival_distribution
        self.service_distribution: Optional[Callable[[], float]] = service_distribution
        """
        If the service_distribution parameter is None, then we will use the Erlang distribution.
        """
        self.num_servers = num_servers
        self.random_seed: int = 42
        self.name: str = name
        self.max_queue_length: float = max_queue_length
        self.sim_time_limit: int = sim_time_limit
        self.bulk_service: Optional[None] = 1 if bulk_service is None else bulk_service
        self.quorum: Optional[None] = 1 if quorum is None else quorum
        self.warm_up_period: int = warm_up_period
        self.LOG_FILE = f"logs/{name.lower().replace(" ", "_")}.txt"

        if lam / (mu * self.num_servers) != rho:
            print(f"Config Error: rho must be equal to lam/(c * mu) lam={lam}, mu={mu}, c={num_servers}",file=sys.stderr)
            sys.exit(1)

        if self.sim_time_limit < self.warm_up_period:
            print(f"Config Error: Warm up period is longer than simulation time!", file=sys.stderr)
            sys.exit(2)

        if self.arrival_distribution is None:
            print(f"Config Error: Arrival distribution is None. Please model as lambda/function pointer", file=sys.stderr)

        """
        No longer an issue. Automatically models the service-time distribution as Erlang.
        """
        #if self.service_distribution is None:
            #print(f"Config Error: Service distribution is None. Please model as a lambda/function pointer", file=sys.stderr)






