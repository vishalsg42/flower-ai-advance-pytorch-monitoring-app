import argparse
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import os

import flwr as fl
import torch
from torch.utils.data import DataLoader
from model_loader import ModelLoader

import utils
from flwr_monitoring import GenericMonitoringStrategy, default_metrics, create_monitoring_tool, aggregate_fit_metrics, aggregate_evaluate_metrics, config


warnings.filterwarnings("ignore")


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    centralized_data = utils.load_centralized_data()
    if toy:
        # use only 10 samples as validation set
        centralized_data = centralized_data.select(range(10))

    val_loader = DataLoader(centralized_data, batch_size=16)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, val_loader)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--toy", action="store_true",
                        help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False")
    parser.add_argument('--server-ip', type=str,
                        default=os.getenv('SERVER_IP', '127.0.0.1'), help="Server IP address")
    parser.add_argument('--server-port', type=int,
                        default=os.getenv('SERVER_PORT', 8080), help="Server port")
    parser.add_argument("--model", type=str, default="efficientnet", choices=[
                        "efficientnet", "alexnet"], help="Use either Efficientnet or Alexnet models. If you want to achieve differential privacy, please use the Alexnet model")
    parser.add_argument("--project-name", type=str, default="my-awesome-project", help="Wandb project name")


    args = parser.parse_args()

    # Load model using the ModelLoader
    model_loader = ModelLoader()
    model = model_loader.load_model()

    # Check if the model is None
    if model is None:
        raise ValueError("ModelLoader returned None. Ensure that load_model() returns a valid model instance.")

    model_parameters = [val.cpu().numpy()
                        for _, val in model.state_dict().items()]

    # Create strategy
    base_strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    server_ip = args.server_ip
    server_port = args.server_port
    wandb_project_name = args.project_name

    # Start Flower server for four rounds of federated learning
    monitoring_tool_instance = create_monitoring_tool(
        tool_name="wandb",
        metrics=default_metrics,
        config={
            "project": wandb_project_name,
        }
    )

    # Wrap the base strategy with the monitoring strategy
    monitoring_strategy = GenericMonitoringStrategy(
        base_strategy, monitoring_tool_instance
    )

    print(f"Starting Flower server at {server_ip}:{server_port}")
    # Start Flower server for four rounds of federated learning
    fl_server = fl.server.start_server(
        server_address=f"{server_ip}:{server_port}",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=monitoring_strategy,
    )


if __name__ == "__main__":
    main()
