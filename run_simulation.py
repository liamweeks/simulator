from queue_config import QueueConfig
from queue_model import QueueModel
from matplotlib import pyplot as plt
import numpy as np
import random
#import test_configs
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor


def get_pi_vector(model):
    """
    Returns the pi-vector for the queue model
    """


    vector = run_simulation(model)
    return [round(elem, 7) for elem in vector]


def create_table(pi_vectors, B):
    """
    Generates the steady-state probabilities for table2-1
    :param pi_vectors:
    :param B: the queue length
    :return:
    """

    prefix = """
    \\begin{table}
    \centering
    \\begin{tabular}{|c|ccc|}\hline
         Number in Queue&  $\\rho=\\frac{1}{6}$&  $\\rho={1}$& $\\rho=2$\\\\\hline\n"""

    for n in range(B+1):
        prefix += f"{n}&{pi_vectors[0][n]}&{pi_vectors[1][n]}&{pi_vectors[2][n]} \\\\\hline\n"


    prefix += """
        \end{tabular}
    \caption{Caption}
    \label{tab:placeholder}
    \end{table}
    """

    return prefix

def create_queue_len_probability_graph(pi_vectors):
    plt.axis([0, 100, 0, 0.3])
    plt.plot(pi_vectors[0], label='ρ=0.1')
    plt.plot(pi_vectors[1], label='ρ=0.5')
    plt.plot(pi_vectors[2], label='ρ=1')

    plt.xlabel('Queue Length')
    plt.ylabel('Probability')
    plt.legend()

    plt.savefig('fig2-2.png', dpi=600, bbox_inches='tight')
    plt.show()


def run_simulation(model: QueueModel):

    """
    Runs a single simulation, given a QueueModel and QueueConfig (a parameter of the QueueModel)
    :param model: The queueing model to be simulated
    :return: Nothing
    """

    model.env.process(model.source())

    for server_id in range(0, model.config.num_servers):
        model.env.process(model.server_loop(server_id))

    model.env.run(until=model.config.sim_time_limit)

    # print(f"Model Name: {model.config.name}")
    # print(f"Pi-Vector:\n{model.pi_vector()}")
    # print(f"Pi-Vector Sum: {sum(model.pi_vector())}")
    # #print(model.recorder.to_frame().describe())
    # #model.recorder.to_csv()
    # #print(f"Pi Vector: {model.recorder.pi_vector()}")
    # #print(f"Pi Vector Sum: {model.recorder.pi_vector_sum()}")
    # model.recorder.quick_stats()
    #
    # # model.recorder.plot_wait_per_packet(window=200)
    # # plt.savefig(f"wait_time_per_packet_{model.config.name.lower().replace(" ", "_")}.png", dpi=300, bbox_inches='tight')
    # fig, (ax_series, ax_ecdf, ax_pi_vec, ax_stats) = plt.subplots(1, 4, figsize=(14, 5))
    # model.recorder.plot_wait_per_packet(window=200, ax=ax_series)
    # model.recorder.plot_wait_ecdf(ax=ax_ecdf)
    # model.recorder.plot_pi_vector(ax=ax_pi_vec)
    # model.recorder.plot_stats(ax=ax_stats)
    #
    # fig.suptitle(model.config.name)
    # fig.tight_layout()
    #
    # graph_file_name = f"wait_time_{model.config.name.lower().replace(' ', '_').replace('.', '')}.png"
    # fig.savefig(graph_file_name, dpi=300, bbox_inches='tight')
    # plt.close(fig)
    return model.pi_vector()

def simulate_models():
    """
    This function is meant to be edited. The configs list is a list of QueueConfigs with the number of
    :return:
    """

    configs = [
        QueueConfig(
            name="Figure2-2-LAM025",
            lam=0.25,
            mu=None,
            num_servers=1,
            arrival_distribution=None, # exponential distribution
            service_distribution=None, # erlang
            rho=3/3,
            B=80,
            sim_time_limit=5e5,
            warm_up_period=5e3,
            b=41,
            quorum=21,
        ),

        QueueConfig(
            name="Figure2-2-LAM125",
            lam=1.25,
            mu=None,
            num_servers=1,
            arrival_distribution=None,  # exponential distribution
            service_distribution=None,  # erlang
            rho=3 / 3,
            B=80,
            sim_time_limit=5e5,
            warm_up_period=5e3,
            b=41,
            quorum=21,
        ),

        QueueConfig(
            name="Figure2-2-LAM25",
            lam=2.5,
            mu=None,
            num_servers=1,
            arrival_distribution=None,  # exponential distribution
            service_distribution=None,  # erlang
            rho=3 / 3,
            B=80,
            sim_time_limit=5e5,
            warm_up_period=5e3,
            b=41,
            quorum=21,
        ),

    ]

    with ThreadPoolExecutor() as executor:
        futures  = [
            executor.submit(get_pi_vector, QueueModel(config)) for config in configs
        ]

        results = [future.result() for future in futures]

        create_queue_len_probability_graph(results)







if __name__ == "__main__":
    start = time.perf_counter()

    simulate_models()

    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=elapsed)}")