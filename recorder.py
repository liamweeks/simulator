from typing import List, Callable
from matplotlib import pyplot as plt
import numpy as np
import simpy

from packet import Packet
import pandas as pd

from queue_config import QueueConfig


class Recorder:

    """
    Records a list of packets through the simulation and exports them to a pandas.DataFrame
    """

    def __init__(self, config: QueueConfig):
        self.records: List[Packet] = []
        self.config: QueueConfig = config

    def collect(self, pkt: Packet):
        """
        Append a packet to the records
        :param pkt: The packet to be appended
        :return: None
        """
        self.records.append(pkt)

    def to_frame(self):
        """
        Create a pandas.DataFrame from the records
        :return:
        """


        df = pd.DataFrame([vars(pkt) for pkt in self.records])
        df['wait_time'] = df['service_start'] - df['arrival']
        return df



    def to_csv(self,name:str='model_output.csv'):
        """
        Export the records as a csv file. The file is placed in the directory specified by `name`
        :param name: The name+path of the csv file.
        :return: None
        """
        self.to_frame().to_csv(name,index=False)

    def quick_stats(self):
        df = self.to_frame()
        print(f"Packets: {len(df)}")
        print(f"Never Departed: {df['departure'].isna().sum()}")
        print(f"Max Occupancy Seen: {self.pi_vector().index.max()}")


    def pi_vector(self):
        """
        Computes the pi vector of the queue model. The pi vector is defined as the steady state probabilities
        of the system being in state n. State n represent that there are n packets in the queue model.
        :return: None
        """

        df = pd.DataFrame(self.records)
        df_length = len(df)
        df = df.iloc[int(0.5 * df_length):]

        events = pd.concat([
            pd.DataFrame({"t" : df['arrival'], "delta" : 1}),
            pd.DataFrame({"t" : df['departure'], "delta" : -1}),
        ]).sort_values("t", kind='stable')

        events["n"] = events['delta'].cumsum()
        left_behind = events.loc[events['delta'] == -1, "n"]
        pi = left_behind.value_counts(normalize=True).sort_index()

        return pi

    def pi_vector_sum(self):
        """
        Sums the elements of the pi vector. The pi vector should ideally sum to 1
        :return:
        """
        return sum(self.pi_vector())

    def plot_wait_ecdf(self, burn_in: float = 0.5, ax=None):
        """
        Plot the empirical CDF of queue wait time (Wq). The ECDF is the clearest
        view because Wq has a probability mass at zero (packets that enter service
        immediately), which appears as a jump at the origin.
        :param burn_in: fraction of records discarded as warm-up transient
        :param ax: optional matplotlib Axes to draw on
        :return: the matplotlib Axes
        """
        df = self.to_frame()
        df = df.iloc[int(burn_in * len(df)):]
        w = df["wait_time"].dropna().sort_values().values
        y = (pd.Series(range(1, len(w) + 1)) / len(w)).values

        if ax is None:
            _, ax = plt.subplots()
        ax.step(w, y, where="post")
        ax.set_xlabel(r"Queue wait time $W_q$")
        ax.set_ylabel(r"$P(W_q \leq t)$")
        ax.set_title("Empirical CDF of queue wait time")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_wait_per_packet(self, window: int = None, ax=None):
        """
        Plot each packet's queue wait time (Wq) against its arrival order. No
        burn-in is applied: this view is meant to expose the warm-up transient
        and any bursts, so you can judge whether steady state is reached and how
        much to discard.
        :param window: optional rolling-mean window to overlay a trend line
        :param ax: optional matplotlib Axes to draw on
        :return: the matplotlib Axes
        """
        df = self.to_frame().sort_values("arrival")
        w = df["wait_time"]

        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(df["id"], w, s=6, alpha=0.4, label="per packet")
        if window:
            ax.plot(df["id"], w.rolling(window, min_periods=1).mean(),
                    color="C1", lw=2, label=f"rolling mean (w={window})")
            ax.legend()
        ax.set_xlabel("Packet (arrival order)")
        ax.set_ylabel(r"Queue wait time $W_q$")
        ax.set_title(f"Queue wait time per packet")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_pi_vector(self, ax=None):
        """
        Plot the steady-state distribution pi(n) — the probability that the system
        holds n packets — as a bar chart. Uses pi_vector(), which already applies
        its own warm-up burn-in.
        :param ax: optional matplotlib Axes to draw on
        :return: the matplotlib Axes
        """
        from matplotlib.ticker import MaxNLocator

        pi = self.pi_vector()

        if ax is None:
            _, ax = plt.subplots()
        ax.bar(pi.index, pi.values, width=0.9)
        ax.set_xlabel(r"Number in system $n$")
        ax.set_ylabel(r"$\pi_n$")
        ax.set_title("Steady-state distribution")
        ax.grid(True, axis="y", alpha=0.3)
        #ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return ax

    def compute_stats(self) -> dict:
        """
        Summary statistics over the full run. No burn-in is applied: Little's law
        is a sample-path identity that holds regardless of warm-up. Returns a dict.
        """
        df = self.to_frame()
        done = df.dropna(subset=["departure", "service_start"])
        if len(done) == 0:
            return {}

        c = self.config.num_servers
        n = len(done)

        def level(plus_col, minus_col):
            ev = pd.concat([
                pd.DataFrame({"t": done[plus_col], "x": 1}),
                pd.DataFrame({"t": done[minus_col], "x": -1}),
            ]).sort_values("t", kind="stable")
            times = ev["t"].to_numpy()
            lvl = ev["x"].cumsum().to_numpy()
            avg = np.sum(lvl[:-1] * np.diff(times)) / (times[-1] - times[0])
            return float(avg), int(lvl.max())

        L, max_sys = level("arrival", "departure")
        Lq, max_q = level("arrival", "service_start")

        T = float(done["departure"].max() - done["arrival"].min())
        lam = n / T
        W = float((done["departure"] - done["arrival"]).mean())
        Wq = float(done["wait_time"].mean())
        svc = done["departure"] - done["service_start"]
        S = float(svc.mean())
        inter = np.diff(np.sort(done["arrival"].to_numpy()))

        return {
            "packets": n,
            "never_departed": int(df["departure"].isna().sum()),
            "never_served": int(df["service_start"].isna().sum()),
            "T": T, "lambda": lam, "mu": 1.0 / S, "rho": (L - Lq) / c,
            "W": W, "Wq": Wq, "S": S, "L": L, "Lq": Lq,
            "lam_W": lam * W, "lam_Wq": lam * Wq,
            "p_no_wait": float((done["wait_time"] <= 1e-12).mean()),
            "cv_service": float(svc.std() / S),
            "cv_interarrival": float(inter.std() / inter.mean()),
            "max_in_system": max_sys, "max_in_queue": max_q,
        }

    def plot_stats(self, ax=None):
        """
        Render compute_stats() as a text panel that slots into a subplot stack.
        :param ax: optional matplotlib Axes to draw on
        :return: the matplotlib Axes
        """
        s = self.compute_stats()
        if ax is None:
            _, ax = plt.subplots()
        ax.axis("off")
        if not s:
            ax.text(0, 1, "No completed packets to summarise.", va="top")
            return ax

        cfg = self.config
        dL = abs(s["L"] - s["lam_W"]) / s["L"] * 100
        dLq = abs(s["Lq"] - s["lam_Wq"]) / s["Lq"] * 100
        lines = [
            f"SIMULATION STATISTICS  ({cfg.name})",
            "-" * 40,
            f"packets completed : {s['packets']:>12d}",
            f"never departed    : {s['never_departed']:>12d}",
            f"never served      : {s['never_served']:>12d}",
            f"observed window T : {s['T']:>12.1f}",
            "",
            "rates                  measured    config",
            f"  lambda          : {s['lambda'] * 10:>10.4f}{cfg.lam:>10.4f}",
            f"  mu              : {s['mu'] * 10:>10.4f}{cfg.mu:>10.4f}",
            f"  rho             : {s['rho']:>10.4f}{cfg.rho:>10.4f}",
            "",
            "means",
            f"  W  (sojourn)    : {s['W']:>10.3f}",
            f"  Wq (wait)       : {s['Wq']:>10.3f}",
            f"  E[S] (service)  : {s['S']:>10.3f}",
            f"  L  (in system)  : {s['L']:>10.3f}",
            f"  Lq (in queue)   : {s['Lq']:>10.3f}",
            "",
            "Little's law            L     lam*W   delta",
            f"  system          : {s['L']:>8.3f}{s['lam_W']:>9.3f}{dL:>7.2f}%",
            f"  queue           : {s['Lq']:>8.3f}{s['lam_Wq']:>9.3f}{dLq:>7.2f}%",
            "",
            "distribution checks",
            f"  P(no wait)      : {s['p_no_wait']:>10.3f}",
            f"  Cv service      : {s['cv_service']:>10.3f}   (M~1, D~0)",
            f"  Cv interarrival : {s['cv_interarrival']:>10.3f}   (M~1)",
            f"  max in system   : {s['max_in_system']:>10d}",
            f"  max in queue    : {s['max_in_queue']:>10d}",
        ]
        ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                family="monospace", fontsize=9, transform=ax.transAxes)
        return ax