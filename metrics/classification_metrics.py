import torch
import torch.nn.functional as F


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ Computes the accuracy of predictions given labels.
    Args:
        preds: (num_data, num_classes) tensor of logits
        labels: (num_data) tensor of labels
    Returns:
        accuracy scalar
    """
    return torch.mean((preds == labels).float()).item()


def label_frequency(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ Computes the frequency of each label.
    Args:
        labels: (num_data) tensor of labels
    Returns:
        (num_classes) tensor mapping indicies to their frequency counts
    """
    counts = torch.bincount(labels)
    padding = (0, num_classes - len(counts))
    return F.pad(counts, padding)


def label_frequency_norm(labels: torch.Tensor) -> torch.Tensor:
    """ Computes the normalized frequency of each label.
    Args:
        Same as label_frequency
    Returns:
        (num_classes) tensor mapping indicies to their normalized frequency counts
    """
    freq = label_frequency(labels)
    return freq / torch.sum(freq)


def kl_divergence(freq_x_norm: torch.Tensor, freq_y_norm: torch.Tensor) -> float:
    """ Computes the KL-divergence between two label frequency distributions.
    Args:
        freq_x_norm: (num_classes) tensor of norm label frequencies
        freq_y_norm: (num_classes) tensor of norm label frequencies
    Returns:
        KL-divergence scalar
    """
    return F.kl_div(freq_x_norm, freq_y_norm)


def kl_divergence_symmetric(freq_x_norm: torch.Tensor, freq_y_norm: torch.Tensor) -> float:
    """ Computes the Jensen-Shannon Divergence between two label frequency distributions.
    Args:
        Same as KL-divergence
    Returns:
        Jensen-Shannon Divergence scalar
    """
    return 0.5 * (kl_divergence(freq_x_norm, freq_y_norm) + kl_divergence(freq_y_norm, freq_x_norm))