import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ Computes the accuracy of predictions given labels.
    Args:
        preds: (num_data, num_classes) tensor of logits
        labels: (num_data) tensor of labels
    Returns:
        accuracy: scalar tensor of accuracy
    """
    return torch.mean((preds == labels).float())


def label_frequency(labels: torch.Tensor) -> torch.Tensor:
    """ Computes the frequency of each label.
    Args:
        labels: (num_data) tensor of labels
    Returns:
        (num_classes) tensor where index i is the frequency of label i
    """
    frequency = torch.bincount(labels)
    return {i: f.item() for i, f in enumerate(frequency)}


def label_frequency_norm(labels: torch.Tensor) -> torch.Tensor:
    """ Computes the normalized frequency of each label.
    Args:
        labels: (num_data) tensor of labels
    Returns:
        (num_classes) tensor where index i is the normalized frequency of label i
    """
    frequency = label_frequency(labels)
    sum = sum(frequency.values())
    return {i: f / sum for i, f in frequency.items()}


