import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ Computes the accuracy of predictions given labels.
    Args:
        preds: (num_data, num_classes) tensor of logits
        labels: (num_data) tensor of labels
    Returns:
        accuracy scalar
    """
    return torch.mean((preds == labels).float()).item()


def label_frequency(labels: torch.Tensor) -> torch.Tensor:
    """ Computes the frequency of each label.
    Args:
        labels: (num_data) tensor of labels
    Returns:
        dict mapping indicies (num_classes) to their frequency counts
    """
    frequency = torch.bincount(labels)
    return {i: f.item() for i, f in enumerate(frequency)}


def label_frequency_norm(labels: torch.Tensor) -> torch.Tensor:
    """ Computes the normalized frequency of each label.
    Args:
        labels: (num_data) tensor of labels
    Returns:
        dict mapping indicies (num_classes) to their normalized frequency counts
    """
    frequency = label_frequency(labels)
    sum = sum(frequency.values())
    return {i: f / sum for i, f in frequency.items()}


