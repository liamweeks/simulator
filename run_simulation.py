from queue_config import QueueConfig
from queue_model import QueueModel
from matplotlib import pyplot as plt
import numpy as np
import random
import test_configs

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

    print(model.recorder.to_frame().describe())
    model.recorder.to_csv()
    print(f"Pi Vector: {model.recorder.pi_vector()}")
    model.recorder.quick_stats()

    # model.recorder.plot_wait_per_packet(window=200)
    # plt.savefig(f"wait_time_per_packet_{model.config.name.lower().replace(" ", "_")}.png", dpi=300, bbox_inches='tight')
    fig, (ax_series, ax_ecdf, ax_pi_vec, ax_stats) = plt.subplots(1, 4, figsize=(14, 5))
    model.recorder.plot_wait_per_packet(window=200, ax=ax_series)
    model.recorder.plot_wait_ecdf(ax=ax_ecdf)
    model.recorder.plot_pi_vector(ax=ax_pi_vec)
    model.recorder.plot_stats(ax=ax_stats)

    fig.suptitle(model.config.name)
    fig.tight_layout()

    graph_file_name = f"wait_time_{model.config.name.lower().replace(' ', '_').replace('.', '')}.png"
    fig.savefig(graph_file_name, dpi=300, bbox_inches='tight')
    plt.close(fig)


def simulate_models():
    """
    This function is meant to be edited. The configs list is a list of QueueConfigs with the number of
    :return:
    """

    configs = [
        QueueConfig(
            name="Table2-1",
            lam=0.5,
            mu=3,
            num_servers=1,
            arrival_distribution=lambda: random.expovariate(0.5),
            service_distribution=lambda: random.expovariate(0.5),
            rho=0.5 / 3,
            max_queue_length=8,
            sim_time_limit=5e4,
            warm_up_period=5e3,
            bulk_service=8,
            quorum=2,
        ),

    ]

    for config in configs: # test_configs.CONFIGS
        run_simulation(QueueModel(config))





if __name__ == "__main__":
    simulate_models()