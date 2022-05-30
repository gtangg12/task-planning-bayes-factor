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


def norm_label_frequency(freq: torch.Tensor) -> torch.Tensor:
    """ Computes the normalized frequency of each label.
    Args:
        (num_classes) tensor of label frequencies
    Returns:
        (num_classes) nomralized label frequencies
    """
    return freq / torch.sum(freq)


def kl_divergence(x_freq: torch.Tensor, y_freq: torch.Tensor) -> float:
    """ Computes the KL-divergence between two label frequency distributions.
    Args:
        freqx: (num_classes) tensor of label frequencies
        freqy: (num_classes) tensor of label frequencies
    Returns:
        KL-divergence scalar
    """
    x_freq = norm_label_frequency(x_freq + 1) # smooth distributions to avoid inifinite KL 
    y_freq = norm_label_frequency(y_freq + 1) 
    return F.kl_div(x_freq.log(), y_freq)


def kl_divergence_symmetric(x_freq: torch.Tensor, y_freq: torch.Tensor) -> float:
    """ Computes the Jensen-Shannon Divergence between two label frequency distributions.
    Args:
        Same as KL-divergence
    Returns:
        Jensen-Shannon Divergence scalar
    """
    return 0.5 * (kl_divergence(x_freq, y_freq) + kl_divergence(y_freq, x_freq))


if __name__ == '__main__':
    a = torch.tensor([23, 76, 580, 33, 0, 38, 0])
    b = torch.tensor([94, 129, 466, 19, 5, 37, 0])
    print(kl_divergence(a, b))