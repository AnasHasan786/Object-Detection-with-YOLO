import os
import matplotlib.pyplot as plt

# Define directories for outputs
OUTPUTS_DIRECTORY = os.path.join("outputs")
IMAGES_DIRECTORY = os.path.join(OUTPUTS_DIRECTORY, "visualizations")

# Ensure output directories exist
os.makedirs(IMAGES_DIRECTORY, exist_ok=True)

def plot_loss_curves(history):
    """
    Plot and save the training and validation loss curves.

    Args:
        history: Training history object from TensorFlow/Keras.
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))

    # Plot loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(IMAGES_DIRECTORY, "loss_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curves plot saved at: {plot_path}")
