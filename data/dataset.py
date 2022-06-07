from data.generate_data import get_states, get_images
from typing import Union, Callable
import numpy as np


class DataLoader:
    def __init__(
        self,
        dataset_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
        dataset_args: Union[tuple, dict],
        batch_size: int,
        post_process: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ) -> None:
        self.obs, self.action = dataset_fn(**dataset_args)
        self.batch_size = batch_size
        self.post_process = post_process
        assert (self.obs.shape[1] % batch_size) == 0
        self.n = 0

    def __len__(self):
        return int(self.obs.shape[1] / self.batch_size)

    def __getitem__(self, index):
        if index < len(self):
            return (
                self.post_process(
                    self.obs[:, index * self.batch_size : (index + 1) * self.batch_size]
                ),
                self.action[:, index * self.batch_size : (index + 1) * self.batch_size],
            )
        else:
            raise IndexError

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            idx = self.n
            self.n += 1
            return self[idx]
        else:
            raise StopIteration
