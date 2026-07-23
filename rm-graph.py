import math
import sys
from algorithm_1 import calculate_pi, construct_pi_vector
import numpy as np
from typing import List, Callable
import matplotlib.pyplot as plt
import monte_carlo_deterministic as monte
from monte_carlo_deterministic import R_saturated

"""
The purpose of this code is to generate the R-M graph (Fig. 8) of the Feng paper.
The graph plots R (average throughput) as a function of M (beam illumination time length)
under deterministic beam hopping
"""

Nu = 128 # users
Nc = 32 # channels
Nm = 3 # multi-packet reception
T = 0.01 # timeslot duration
Tp = 0.1 # illumination period
I = 20 # buffer capacity
P_TR = 0.4 # transmission probability
#M = 8 # illuminated timeslots
#N = 2 # unilluminated timeslots
#LAMBDA = 10 # packet arrival rate
P_ON = 0.5
P_S = 0.8

def theoretical_R(mu, lam, N_u=Nu, T=T):
    """
    Calculates the theoretical average throughput of the queue beam hopping model
    :param N_u: The number users in a ground cell
    :param lam: The packet arrival rate
    :param T: The duration of each slot
    :return: The theoretical average throughput
    """
    return N_u * min(lam * T, mu * T)

def theoretical_mu(p_s_by_slot,M, N, T=T, p_tr=P_TR):
    """
    Calculates the theoretical packet service rate
    :param M: The beam illumination time
    :param p_tr: The probability of transmission
    :param T: The duration of each slot
    :param N: Non-illumination length
    :return: The theoretical service rate
    """
    mu: float = 0

    for t in range(1, M + 1):
        summand = p_tr * p_s_by_slot[t] / ((M + N) * T)
        mu += summand

    return mu

def ps_t(t:int, Nm: int, Nu: int, Nc: int, p_tr: float, pi_vector, M:int):
    """
    Calculates the probability of a successful transmission at time t
    :param t: The timeslot where you want to calculate the probability of a successful transmission
    :param Nm: The Multi-Packet Reception Capability (How many messages you can deal with at once)
    :param Nu: The number of users in a ground cell
    :param Nc: The number of frequency channels for communication
    :param p_tr: The probability of a transmission
    :return: The probability of a successful transmission at time t
    """

    p_s_t: float = 0



    for n in range(Nm):
        #pi_vector = construct_pi_vector(I, LAMBDA, t, P_ON, P_TR, P_S)
        p_non_empty = pex_t(t,pi_vector=pi_vector, M=M)
        summand = math.comb(Nu-1, n) * math.pow(p_non_empty * P_TR / Nc, n) * math.pow(1 - (p_non_empty * p_tr / Nc), Nu - 1 - n)
        p_s_t += summand

    return p_s_t

def state_to_index(t: int, i: int, I=I):
    """
    Used to index from the state space of the 2D Markov chain to
    an index
    :param t: The time-slot of the illumination period
    :param i: The queue length
    :param I: The capacity of the queue
    :return: the 1D index of the state
    """
    return (t - 1) * (I + 1) + i


def pex_t(t: int, pi_vector, M, I=I):
    """
    Calculates the probability of a non-empty queue given
    a time-slot in the illumination period. (E32 in Feng)
    :param I: The capacity of the buffer
    :param M: The number of illumination slots
    :param t: The time-slot in the illumination period
    :param pi_vector: The steady-state probabilities
    :return:
    """

    #pi_vector = construct_pi_vector(I, LAMBDA, t, P_ON, P_TR, P_S)
    if not 0 < t <= M:
        print(f"Error: t={t} must be within the illumination period (M={M})", file=sys.stderr)

    index = state_to_index(t, 0)
    try:
        p_empty_at_t = pi_vector[index]
    except IndexError:
        print(f"Error: Tried to access index {index}, |pi_vec| = {len(pi_vector)}", file=sys.stderr)
        raise IndexError

    summation = 0

    for i in range(I + 1):
        summation += pi_vector[state_to_index(t, i, I)]

    return 1 - (p_empty_at_t / summation)


def build_phase_matrix(t:int, M, I=I):
    """
    (I+1) x (I+1) queue transition matrix for A_t at phase t
    :param t:
    :param I:
    :param M:
    :return:
    """

    A = np.zeros((I+1, I+1))
    illuminated = (t <= M)
    for i in range(I + 1):
        for j in range(I + 1):
            A[i, j] = ...

    return A


def build_P(p0, p1, p_s_by_slot, M, N, I=I):
    """
    Returns the (M+N)(I+1) x (M+N)(I+1) one step transition P = p_{j, k}
    :param I:
    :param M:
    :param N:
    :param p0:
    :param p1:
    :param p_s_by_slot:
    :param p_tr:
    :return:
    """
    Ttot = M + N
    n = Ttot * (I+1)
    P = np.zeros((n, n))

    def add(t, i, t_next, i_next, prob):
        P[state_to_index(t, i), state_to_index(t_next, i_next)] += prob

    for t in range(1, Ttot + 1):
        t_next = t + 1 if t < Ttot else 1
        illuminated = (t <= M)
        beta = P_TR * p_s_by_slot[t] if illuminated else 0.0 # successful transmission

        for i in range(I + 1):
            if i == 0:
                add(t, 0, t_next, 0, p0)
                add(t, 0, t_next, 1, p1)
            elif i <  I:
                if illuminated:
                    add(t, i, t_next, i - 1, p0 * beta)
                    add(t, i, t_next, i, p0 * (1 - beta) + p1 * beta)
                    add(t, i, t_next, i + 1, p1 * (1 - beta))
                else:
                    add(t, i, t_next, i, p0)
                    add(t, i, t_next, i + 1, p1)
            else:
                if illuminated:
                    add(t, I, t_next, I - 1, p0 * beta)
                    add(t, I, t_next, I, 1 - p0 * beta)
                else:
                    add(t, I, t_next, I,  1.0)

    return P

def solve_pi(P):
    """
    pi * P = pi and sum(pi) = 1
    :param P:
    :return:
    """

    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def main(lam, M, N):
    p0 = 1 - P_ON
    p1 = P_ON
    ps_by_slot = {t: 1.0 for t in range(1, M + 1)}

    for _ in range(100):
        P = build_P(p0, p1, ps_by_slot, M, N)
        pi = solve_pi(P)
        new = {t: ps_t(t, Nm, Nu, Nc, P_TR, pi, M) for t in range(1, M + 1)}

        if max(abs(new[t] - ps_by_slot[t]) for t in new) < 1e-9:
            ps_by_slot = new
            break
        ps_by_slot = new

    mu = theoretical_mu(ps_by_slot, M=M, N=N)
    r = theoretical_R(mu,lam=lam)

    return r

if __name__ == "__main__":
    lambdas = [1, 5, 10, 15, 20]
    illumination_periods = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    simulated = dict.fromkeys(lambdas)

    results = dict.fromkeys(lambdas)

    print(results)

    for l in lambdas:
        results[l] = []
        simulated[l] = []
        for m in illumination_periods:
            n = 10 - m
            r = main(lam=l,M=m,N=n)

            mean, hw = monte.simulate_system_ci(m, n, arrival_prob=l * T, reps=5, n_slots=400, warmup=20)
            ra = R_saturated(m, n)

            print(f"L={l}, M={m}: Theoretical R: {r} Simulated R: {mean} ({abs(mean - ra) / ra})")

            results[l].append(r)
            simulated[l].append(mean)

    print(results)

    for l in lambdas:
        plt.plot(illumination_periods, results[l], marker="o", label=fr'$\lambda={l}$')
        plt.plot(illumination_periods, simulated[l], marker='.', label=fr'$\lambda={l}$')

    plt.xticks(illumination_periods)
    plt.xlabel("Illumination Length (M)")
    plt.ylabel("Throughput (R)")
    plt.legend()
    plt.grid()
    plt.title("Average Throughput vs. Beam Illumination Time")
    plt.savefig('rm_graph.png',dpi=600)

