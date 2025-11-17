# FashionMNIST Image Classification

This project contains a PyTorch implementation for training a convolutional neural network (CNN) to classify images from the FashionMNIST dataset.

## File Description

-   `vision_model.py`: The main script that handles data loading, model definition, training, and evaluation.
-   `requirements.txt`: A list of Python dependencies required to run the project.

## Model Architecture

The script implements a CNN model named `fashion_minst_model_v2`, which is based on the TinyVGG architecture. The model consists of two convolutional blocks followed by a classifier head.

-   **Convolutional Block 1:**
    -   `nn.Conv2d` (in_channels=1, out_channels=10, kernel_size=3, padding=1)
    -   `nn.ReLU`
    -   `nn.Conv2d` (in_channels=10, out_channels=10, kernel_size=3, padding=1)
    -   `nn.ReLU`
    -   `nn.MaxPool2d` (kernel_size=2)
-   **Convolutional Block 2:**
    -   `nn.Conv2d` (in_channels=10, out_channels=10, kernel_size=3, padding=1)
    -   `nn.ReLU`
    -   `nn.Conv2d` (in_channels=10, out_channels=10, kernel_size=3, padding=1)
    -   `nn.ReLU`
    -   `nn.MaxPool2d` (kernel_size=2)
-   **Classifier:**
    -   `nn.Flatten`
    -   `nn.Linear` (in_features=490, out_features=10)

## Setup

1.  **Clone the repository or download the project files.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To train the model, simply run the `vision_model.py` script:

```bash
python vision_model.py
```

The script will download the FashionMNIST dataset, train the model for 5 epochs, and print the training progress and final evaluation results to the console.

## Performance

After training for 5 epochs, the model achieved the following performance on the test set:

-   **Test Accuracy:** 88.83%
-   **Total Training Time:** Approximately 44 seconds on an Apple MPS device.

## Code Summary

The `vision_model.py` script is organized into the following key components:

-   **Data Loading:** The script uses `torchvision.datasets.FashionMNIST` to download and load the training and testing data. `DataLoader` is used to create data batches.
-   **Model Definitions:** Three models are defined:
    -   `fashion_minst_model_v0`: A simple linear model.
    -   `fashion_minst_model_v1`: A linear model with ReLU activation.
    -   `fashion_minst_model_v2`: The main CNN model used for training.
-   **Helper Functions:**
    -   `get_device()`: Detects and returns the available device (CUDA, MPS, or CPU).
    -   `accuracy_fn()`: Calculates the classification accuracy.
    -   `eval_model()`: Evaluates the model on a given dataset.
-   **Training and Testing Functions:**
    -   `train_step()`: Performs a single training step (forward pass, loss calculation, backward pass, and optimizer step).
    -   `test_step()`: Performs a single testing step (evaluation on the test set).
-   **Prediction Function:**
    -   `make_predictions()`: A function to make predictions on a list of data samples.
-   **Main Execution Block:** The `if __name__ == "__main__":` block orchestrates the entire process: model instantiation, training loop, and final evaluation.
