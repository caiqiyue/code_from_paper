"""Lightweight dataset wrappers used throughout the PrE-Text pipeline."""

from torch.utils.data import Dataset


class ListDataset(Dataset):
    """Wrap a Python list of raw text strings as a PyTorch dataset."""

    def __init__(self, text_list):
        """Store the text samples that will be served by index."""
        self.text_list = text_list

    def __len__(self):
        """Return the number of text samples in the dataset."""
        return len(self.text_list)

    def __getitem__(self, idx):
        """Return one raw text sample by integer index."""
        return self.text_list[idx]


class MatrixDataset(Dataset):
    """Wrap token matrices so DataLoader can iterate over candidate sequences."""

    def __init__(self, inputs):
        """Store token ids and attention masks for candidate parent texts."""
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]

    def __len__(self):
        """Return the number of tokenized sequences in the matrix."""
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        """Return one tokenized sequence and its mask as batch-shaped tensors."""
        return {
            "input_ids": self.input_ids[idx, :][None, :],
            "attention_mask": self.attention_mask[idx, :][None, :],
        }
