import numpy as np
from abc import ABC, abstractmethod

class Dataset(ABC):
    """Abstract dataset protocol used by the task loaders."""

    @abstractmethod
    def __init__(self):
        """Initialize the Dataset instance."""
        pass
    
    @abstractmethod
    def __getitem__(self):
        """Return the sample at the requested index."""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return the number of available samples in the current split."""
        pass


class DataLoader:
    """Minimal iterator that batches task datasets for training."""
    def __init__(self, data, batch_size=32, shuffle=True):
        """Initialize the DataLoader instance."""
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.current_index = 0

    def __iter__(self):
        """Reset the iterator state and optionally reshuffle sample indices."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0
        return self

    def __next__(self):
        """Return the next mini-batch from the wrapped dataset."""
        if self.current_index >= len(self.data):
            raise StopIteration

        batch_indices = self.indices[self.current_index : self.current_index + self.batch_size]
        batch_data = [self.data[int(i)] for i in batch_indices]

        if isinstance(batch_data[0], tuple):
            batch_data = tuple(zip(*batch_data))
        else:
            batch_data = np.array(batch_data)

        self.current_index += self.batch_size

        return batch_data
