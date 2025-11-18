"""FLfraud: A Flower / PyTorch app."""

import torch
import wandb
import warnings
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from FedCAD.task import Net, load_data, test

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Initialize wandb for server-side logging
    wandb.init(
        project="FedCAD",
        name="federated_server",
        config={
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "lr": lr,
        },
        mode="online",  # Only sync wandb runs to cloud, don't save locally
        settings=wandb.Settings(
            _disable_stats=True,
            _disable_meta=True,
        )
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Evaluate final model on test data
    print("\nEvaluating final model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    final_model = Net()
    state_dict = result.arrays.to_torch_state_dict()
    final_model.load_state_dict(state_dict)
    
    # Load test data (using partition 0 for centralized test set)
    _, testloader = load_data(0, 1)
    test_loss, test_acc = test(final_model, testloader, device)
    
    # Log final metrics
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_accuracy": test_acc,
    })
    
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Save final model to disk
    print("\nSaving final model to disk...")
    torch.save(state_dict, "final_model.pt")
    
    wandb.finish()
