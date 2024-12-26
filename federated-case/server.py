import flwr as fl
import utils
from flwr.server.strategy import FedAvg
from logging import *


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            model = utils.load_model()
            # Save aggregated_ndarrays to disk
            if server_round % 5 == 0 or server_round == 2:
                print(f"Saving round {server_round} model")
                model.set_weights(aggregated_ndarrays)
                model.save(f"model-round{server_round}.keras")

        return aggregated_parameters, aggregated_metrics



    
fl.server.start_server(
    server_address="192.168.86.21:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy= SaveModelStrategy(
        min_available_clients= 4
    )
    )


