# Core
import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Torchvision
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import GaussianBlur
from torch.utils.data import DataLoader

# Plotting
import matplotlib.pyplot as plt


############### GENERAL ###############


def plot_training_curves(loss_curve, accuracy_curve, epoch_markers, epochs):
    """
    Plots loss and accuracy curves for training monitoring.

    Args:
        loss_curve (list): List of recorded loss values.
        accuracy_curve (list): List of recorded accuracy values.
        epoch_markers (list): Indices where epochs ended.
        epochs (int): Number of epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Training Curves")

    # Loss curve
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_xticks(epoch_markers, tuple(range(1, epochs + 1)))
    ax1.plot(loss_curve, label="Loss", color="C0")
    ax1.scatter(epoch_markers, [loss_curve[i] for i in epoch_markers], color="C0")

    # Accuracy curve
    ax2.set_title("Accuracy on Test Dataset (%)")
    ax2.set_xlabel("Epoch")
    ax2.plot(range(1, epochs + 1), accuracy_curve, label="Accuracy", color="C1", marker="o")
    # ax2.hlines(baseline, 1, epochs, linestyle="dashed", color="gray", alpha=0.5)

    fig.tight_layout()
    plt.show()


def plot_compare_training_curves(loss_bn, loss_no_bn, acc_bn, acc_no_bn, epochs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("BatchNorm vs No BatchNorm – Training Curves")

    # ---- Loss ----
    ax1.plot(range(1, epochs + 1), loss_no_bn, label="No BN", color="C0", marker="o")
    ax1.plot(range(1, epochs + 1), loss_bn, label="With BN", color="C1", marker="o")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # ---- Accuracy ----
    ax2.plot(range(1, epochs + 1), acc_no_bn, label="No BN", color="C0", marker="o")
    ax2.plot(range(1, epochs + 1), acc_bn, label="With BN", color="C1", marker="o")
    ax2.set_title("Test Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    fig.tight_layout()
    plt.show()


def plot_compare_training_eval_curves(acc_eval, acc_no_eval, epochs):

    plt.figure(figsize=(12, 4))
    plt.title("BatchNorm: Eval Mode vs Train Mode (Test Accuracy)")

    plt.plot(range(1, epochs + 1), acc_no_eval, label="No Eval", color="C0", marker="o")
    plt.plot(range(1, epochs + 1), acc_eval, label="Eval", color="C1", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.show()


def plot_compare_training_batch_curves(loss_bn, loss_no_bn, acc_bn, acc_no_bn, batch_number):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("BatchNorm vs No BatchNorm – Training Curves Across Batch Sizes")

    # ---- Loss ----
    ax1.plot(range(1, batch_number + 1), loss_no_bn, label="No BN", color="C0", marker="o")
    ax1.plot(range(1, batch_number + 1), loss_bn, label="With BN", color="C1", marker="o")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Batch Number")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # ---- Accuracy ----
    ax2.plot(range(1, batch_number + 1), acc_no_bn, label="No BN", color="C0", marker="o")
    ax2.plot(range(1, batch_number + 1), acc_bn, label="With BN", color="C1", marker="o")
    ax2.set_title("Test Accuracy (%)")
    ax2.set_xlabel("Batch Number")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    fig.tight_layout()
    plt.show()


def train_model(model, optimizer, criterion, train_loader, test_loader, epochs, eval_status=True):
    train_losses, test_accuracies = [], []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc="Training batches"):
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Average loss
        train_losses.append(running_loss / len(train_loader))

        # Evaluate accuracy
        if eval_status:
            model.eval()
        else:
            pass
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        test_accuracies.append(100 * correct / total)

    return train_losses, test_accuracies


############### MNIST POSTER ###############


def show_fixed_predictions(model, images, labels):
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig = plt.figure(figsize=(12, 10))
    rows, cols = 5, 6  # 5x6 = 30 images
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f"P:{predicted[i].item()}\nT:{labels[i].item()}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_first_layer_weights(model, input_shape=(1, 28, 28), nrow=8, ncol=8, figsize=(8, 8)):
    """
    Visualizes the first linear layer weights of a fully-connected network trained on 2D image data.
    """
    # Find the first Linear layer
    linear = None
    for layer in model.net:
        if isinstance(layer, torch.nn.Linear):
            linear = layer
            break

    if linear is None:
        raise ValueError("No Linear layer found in the model.")

    # Extract weights (shape: [out_features, in_features])
    W = linear.weight.detach().cpu().clone()

    # Normalize weights for visualization
    W = W - W.min()
    W = W / W.max()

    # Reshape weights to image format
    W = W.reshape(-1, *input_shape)  # (num_neurons, 1, 28, 28)

    # Show only first nrow*ncol filters
    W = W[:nrow * ncol]

    # Create grid
    grid = torchvision.utils.make_grid(W, nrow=nrow, normalize=False, pad_value=1.0)
    grid = grid.permute(1, 2, 0)

    # Plot grid
    plt.figure(figsize=figsize)
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    plt.title("First Linear Layer Weights (as Images)")
    plt.show()


def visualize_linear1_weights_2d(network, input_shape, nrow=4, ncol=4, figsize=(8, 8)):
    """
    Visualizes the first linear layer weights, assuming the network processes 2D images.

    Args:
        network (torch.nn.Module): The neural network with a MLP whose first layer will be visualized.
        input_shape (tuple): Shape of the input to the linear layer (C, H, W).
        nrow (int): Number of filters per row in the grid.
        ncol (int): Number of filters per column in the grid.
        figsize (tuple): Size of the matplotlib figure.
    """
    # Find first linear layer
    linear = None
    for i, layer in enumerate(network.net):
        if isinstance(layer, torch.nn.Linear):
            linear = layer
            break
    if linear is None:
        raise ValueError("No linear layer found in the network.")

    # Extract and normalize weights
    W = linear.weight.cpu().detach().clone()
    W = W - W.min()
    W = W / W.max()
    W = W.reshape(-1, *input_shape)

    if W.shape[0] > nrow * ncol:
        print(f"Warning: Layer has {W.shape[0]} features, but only first {nrow * ncol} will be displayed.")

    W = W[:nrow * ncol]

    # Create grids for visualization
    grid = torchvision.utils.make_grid(W, nrow=nrow).permute(1, 2, 0)

    # Plot the results
    _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(grid)
    ax.axis('off')

    plt.tight_layout()
    plt.show()


def get_relu_activations(model, data_loader):
    activations = []

    def hook(module, input, output):
        activations.append(output.detach())

    relu_layer = None
    for layer in model.net:
        if isinstance(layer, nn.ReLU):
            relu_layer = layer
            break

    handle = relu_layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            model(x)

    handle.remove()
    return torch.cat(activations, dim=0)


def count_dead_neurons(activations, threshold=0.99):
    if activations.dim() == 2:
        zero_fraction = (activations == 0).float().mean(dim=0)
    elif activations.dim() == 4:
        zero_fraction = (activations == 0).float().mean(dim=(0, 2, 3))
    else:
        raise ValueError("Unsupported activation shape")

    dead_neurons = (zero_fraction > threshold).sum().item()
    avg_zero_rate = zero_fraction.mean().item()
    return dead_neurons, avg_zero_rate


def dead_neuron_plot(dead_no_bn, avg_zero_no_bn, dead_bn, avg_zero_bn):

  fig, axes = plt.subplots(1, 2, figsize=(10, 4))

  # ---- Left: Dead Neurons ----
  axes[0].bar("No BN", dead_no_bn, color="C0")
  axes[0].bar("With BN", dead_bn, color="C1")
  axes[0].set_title("Dead Neurons")
  axes[0].set_ylabel("Count")

  # ---- Right: Average Zero Rate ----
  axes[1].bar("No BN", avg_zero_no_bn, color="C0")
  axes[1].bar("With BN", avg_zero_bn, color="C1")
  axes[1].set_title("Average Zero Activation Rate")
  axes[1].set_ylabel("Zero Rate")

  plt.tight_layout()
  plt.show()


############### CIFAR10 POSTER ###############


def total_variation_loss(img):
    """
    Computes total variation loss for smoothness regularization.
    """
    tv_loss = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
              torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    return tv_loss


def generate_class_image(net, device, class_idx, iterations=1000, lr=0.001,
                         tv_weight=0.000185, blur_every=50):
    """
    Generates an image that maximally activates the specified class in the network.

    Args:
        net (torch.nn.Module): The trained neural network.
        device (torch.device): Device to run the generation on.
        class_idx (int): Index of the class to visualize.
        iterations (int): Number of optimization steps.
        lr (float): Learning rate.
        tv_weight (float): Weight for total variation loss.
        blur_every (int): Frequency of applying Gaussian blur for regularization.

    Returns:
        torch.Tensor: Generated class image (normalized, shape: H x W x C).
    """
    img = torch.randn((1, 3, 32, 32), requires_grad=True, device=device)
    optimizer = torch.optim.AdamW([img], lr=lr)
    blur = GaussianBlur(kernel_size=3, sigma=1)

    for i in range(iterations):
        optimizer.zero_grad()

        # Apply blur every `blur_every` iterations
        if i % blur_every == 0:
            with torch.no_grad():
                img.data = blur(img.data)

        out = net(img)
        class_loss = -out[0, class_idx] + out[0, :].mean()
        tv_loss = total_variation_loss(img) * tv_weight
        loss = class_loss + tv_loss

        loss.backward()
        optimizer.step()

        img.data = img.data.clamp(-3, 3)

    # Normalize image for visualization
    img = img.detach().cpu()
    img = img - img.min()
    img = img / img.max()
    img = img.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    return img


def generate_images_for_all_classes(net, device, class_names,
                                    iterations=1050, lr=0.0015, show=True):
    """
    Generates and optionally displays class-activating images for each class in the model.

    Args:
        net (torch.nn.Module): Trained model.
        device (torch.device): CPU or GPU.
        class_names (list): List of class names corresponding to indices.
        iterations (int): Number of optimization iterations per class.
        lr (float): Learning rate.
        show (bool): Whether to display images with matplotlib.

    Returns:
        list: List of generated class images as torch.Tensor.
    """
    net.to(device)
    images = []

    for idx, class_name in enumerate(class_names):
        print(f'Generating image for class: {class_name}')
        img = generate_class_image(net, device, idx, iterations=iterations, lr=lr)
        images.append(img)

        if show:
            plt.figure(figsize=(2, 2))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'{class_name}')
            plt.show()

    return images


def get_all_relu_activations(model, data_loader):
    activations = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx].append(output.detach())
        return hook

    handles = []
    for idx, layer in enumerate(model.net):
        if isinstance(layer, nn.ReLU):
            activations[idx] = []
            handles.append(layer.register_forward_hook(make_hook(idx)))

    if not activations:
        raise ValueError("No ReLU layers found in model")

    model.eval()
    with torch.no_grad():
        for x, _ in data_loader:
            model(x)

    for h in handles:
        h.remove()

    # Concatenate batches per layer
    for idx in activations:
        activations[idx] = torch.cat(activations[idx], dim=0)

    return activations
