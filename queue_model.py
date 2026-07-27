import random

from packet import Packet
from queue_config import QueueConfig
import simpy
import math
import numpy as np
import sys

from recorder import Recorder
import pandas as pd


class QueueModel:
    """
    The QueueModel class is the heart of the simulator. Given a QueueConfig, it can simulate the queue. The QueueModel
    is generic enough to support G/G/(inf | c)/(inf | k) models.
    """
    def __init__(self, config: QueueConfig):
        self.env: simpy.Environment = simpy.Environment()
        self.config: QueueConfig = config
        self.servers: simpy.Resource = simpy.Resource(self.env, capacity=self.config.num_servers)
        self.queue: simpy.Store = simpy.Store(self.env) if self.config.max_queue_length == math.inf else simpy.Store(self.env, self.config.max_queue_length)
        self._stop_simulation: simpy.Event = self.env.event()
        self.recorder: Recorder = Recorder(self.config)
        self.rng = np.random.default_rng(self.config.random_seed)
        self.n_in_system: int = 0
        self.n_in_server: int = 0
        self.departure_states: list[tuple[float, int]] = []


    def pi_vector(self):
        """
        Post-Departure Epoch Pi-Vector
        :return:
        """
        warm_up_period = self.config.warm_up_period
        states = [n for (t, n) in self.departure_states if t >= warm_up_period]

        if not states:
            return pd.Series(dtype=float)
        else:
            pi = pd.Series(states).value_counts(normalize=True).sort_index()
            # return the pi_n values for [0,B]
            pi = pi.reindex(range(0, self.config.B + 1), fill_value=0.0)
            return list(pi)

    def get_queue_length(self)-> int:
        """
        Returns the number of packets in the queue
        """
        return len(self.queue.items)

    def get_system_capacity(self) -> int:
        """
        Returns the number of packets in the system. The system capacity is calculated as the
        number of packets in the queue + server
        :return:
        """
        return self.n_in_system + self.n_in_server

    def erlang(self, n_in_queue: int):
        """
        Simulates the Erlang distribution for Dr. Gai's Queue models
        :param n_in_queue: The number of packets in the queue.
        :return:
        """
        k = max(1, n_in_queue)
        return self.rng.gamma(shape=k, scale=1.0 / self.config.mu)



    def source(self):
        """
        Models the arrival of packets using the specified arrival distribution and places them in the queue
        :return: A generator
        """

        packets_arrived: int = 0

        # while packets_arrived < self.config.packet_limit:
        #     """
        #     Wait for packet to arrive
        #     """
        #     interval_time = self.config.arrival_distribution()
        #     yield self.env.timeout(interval_time)
        #     packets_arrived += 1
        #
        #     pkt = Packet(id=packets_arrived, arrival=self.env.now)
        #
        #     # print(f"Incoming: {pkt}")
        #
        #     yield self.queue.put(pkt) # wait to be put in the queue if it is full

        dropped_packets: int = 0

        while packets_arrived < self.config.packet_limit:
            """
            Process packet arrival.
            
            - If there is space in the queue, enqueue the arriving packet
            - If the queue is full, then the packet is dropped
            """
            interarrival_time = self.config.arrival_distribution()
            yield self.env.timeout(interarrival_time)
            packets_arrived += 1

            packet = Packet(id=packets_arrived, arrival=self.env.now)

            if self.get_queue_length() < self.config.B:
                self.n_in_system += 1
                yield self.queue.put(packet)
            else:
                # no more space left:
                dropped_packets += 1





    def server_loop(self, server_id: int):
        """
        Models a single server to process packets. The server will wait for `batch_size` packets,
        pull the batch, and then serve them together simultaneously. For single service, the default batch size is 1
        :return: None
        """

        while True:

            first = yield self.queue.get()
            batch = [first]
            config_batch_size = self.config.b
            config_quorum = self.config.quorum

            """
            Wait for the batch to fill up, so the packets can be served.
            
            - The first while loop fills up to the quorum, the minimum amount of packets to start the service
            - The second while loop will get any remaining packets (if there are any)
            - Serve the packets.
            """

            while len(batch) < config_quorum:
                incoming_packet = yield self.queue.get()
                batch.append(incoming_packet)

            while (len(batch) < config_batch_size) and (0 < len(self.queue.items)):
                incoming_packet = yield self.queue.get()
                batch.append(incoming_packet)


            """
            Serve the packet(s). Serves one packet if using the single-server model, multiple packets if using the bulk service model
            """

            if self.config.service_distribution is None:
                """
                Model the service time as an Erlang distribution. All the packets in the group are served at once.
                """
                # j = len(batch)
                #
                # if not (self.config.quorum <= j <= self.config.b):
                #     print(f"Simulator Error: j = {j}. Bounds are [{self.config.quorum},{self.config.b}]",file=sys.stderr)
                #     sys.exit(1)
                #
                # group_service_time = self.erlang(j)
                # start_group_service = self.env.now

                """
                Model the service time as a hyperexponential distribution. All the packets in a group are served at once
                """

                # j = len(batch)
                # sigma1 = 0.4
                # sigma2 = 0.6
                # start_group_service = self.env.now
                # group_service_time: float = random.expovariate(2) if random.random() < sigma1 else random.expovariate(3)

                mu1=2
                mu2=3
                sigma1 = 0.4
                sigma2 = 0.6
                start_group_service = self.env.now
                j=len(batch)
                rates = (mu1, mu2)
                group_service_time = sum(
                    random.expovariate(mu1 if self.rng.random() < sigma1 else mu2)
                    for _ in range(j)
                )



                for packet in batch:
                    # group service => all packets start service at the same instant, and have the same service time.
                    packet.service_start = start_group_service
                    packet.group_size = j

                self.n_in_system -= len(batch)
                self.n_in_server = j
                yield self.env.timeout(group_service_time)
                self.n_in_server = 0

                self.departure_states.append((self.env.now, len(self.queue.items)))

                group_departure_time = self.env.now
                for packet in batch:
                    # record service time
                    packet.departure = group_departure_time
                    self.recorder.collect(packet)



            else:
                """
                Call whatever function is specified to model the service-time
                """

                service_time = self.config.service_distribution()

                # record the start of the service
                start_of_service = self.env.now
                for pkt in batch:
                    pkt.service_start = start_of_service

                yield self.env.timeout(service_time)

                # record the departure time of the packet
                batch_departure_time = self.env.now
                for pkt in batch:
                    pkt.departure = batch_departure_time


                # print("Packets in Batch:", end='\n\t')
                # for pkt in batch:
                #     print(f"{pkt}", end="\n\t")
                # print()

                for pkt in batch:
                    self.recorder.collect(pkt)
