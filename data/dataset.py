from data.generate_data import get_states, get_images
from typing import Union, Callable
import numpy as np


class DataLoader:
    def __init__(
        self,
        dataset_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
        dataset_args: dict,
        batch_size: int,
        random_seed: int,
        post_process: Callable[[np.ndarray], np.ndarray] = lambda x: x,
    ) -> None:
        """ This is a wrapper class for the datasets that enables loading data batches.

        Args:
            dataset_fn: A function that returns a dataset of observation action sequences.
            dataset_args: The keyword arguments for dataset_fn.
            batch_size: The batch size for loading.
            random_seed: The random seed for generating the dataset.
            post_process: A function to transform the observations, i.e. mixing matrix for mixed-state experiments.
        """
        np.random.seed(random_seed)
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
