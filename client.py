import argparse
import warnings
from collections import OrderedDict
import time  # Import the time module

# import datasets
import os
import flwr as fl
import torch
from torch.utils.data import DataLoader
# from data_loader import DataClientLoader
from client_data_loader import DataClientLoader
from model_loader import ModelLoader

import utils

warnings.filterwarnings("ignore")


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset,  # pass trainset explicitly
        testset,   # pass testset explicitly
        # data_loader: DataClientLoader,
        model_loader: ModelLoader,
        device: torch.device,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.model_loader = model_loader
        self.validation_split = validation_split
        self.model = self.model_loader.load_model()
        self.trainset = trainset  # assign trainset
        self.testset = testset    # assign testset

    def set_parameters(self, parameters):
        """Loads a alexnet or efficientnet model and replaces it parameters with the
        ones given."""

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        print("Fitting model on client side ...")

        # Record the start time for the round
        round_start_time = time.time()

        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Split trainset into training and validation sets
        train_valid_split = self.trainset.train_test_split(test_size=self.validation_split, seed=42)
        trainset = train_valid_split["train"]
        valset = train_valid_split["test"]

        # Create data loaders
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size)

        # Train the model
        results = utils.train(self.model, train_loader, val_loader, epochs, self.device)

        # Get updated model parameters
        parameters_prime = utils.get_model_params(self.model)
        num_examples_train = len(trainset)

        # Calculate the round time
        round_end_time = time.time()
        round_time = round_end_time - round_start_time
        print(f"Round time: {round_time:.2f} seconds")

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        print("Evaluating model on client side ...")
        # Record the start time for the evaluation round
        round_start_time = time.time()

        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Create test data loader
        test_loader = DataLoader(self.testset, batch_size=16)

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = utils.test(self.model, test_loader, steps, self.device)

        # Calculate the round time
        round_end_time = time.time()
        round_time = round_end_time - round_start_time
        print(f"Evaluation round time: {round_time:.2f} seconds")

        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def main() -> None:
    # Start time of the entire script
    start_time = time.time()

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to quicky run the client using only 10 datasamples. \
        Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--use-cuda",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use GPU. Default: False",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. \
             If you want to achieve differential privacy, please use the Alexnet model",
    )
    parser.add_argument(
        '--server-ip', type=str,
        default=os.getenv('SERVER_IP', '0.0.0.0'),
        help="Server IP address"
    )
    parser.add_argument(
        '--server-port',
        type=str,
        default=os.getenv('SERVER_PORT', '8080'),
        help="Server port"
    )

    args = parser.parse_args()
    client_id = args.client_id

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    # TODO refactor this to use the DataClientLoader
    data_loader = DataClientLoader(client_id)
    trainset, testset = data_loader.load_data()

    # Load model using the ModelLoader
    model_loader = ModelLoader()

    server_ip = args.server_ip
    server_port = args.server_port

    server_ip = args.server_ip
    server_port = args.server_port
    # Start Flower client
    client = CifarClient(trainset, testset, model_loader, device).to_client()

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}", client=client)

    # End time of the entire script
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")



if __name__ == "__main__":
    main()
