from data.multi_body import get_dataset, render
import numpy as np


def get_images(
    batch_size: int,
    length: int = 1000,
    chunks: int = 20,
    steps: int = 1,
    interventions: list[int] = [[]],
):
    """ Generates an image dataset (Note that this is memory heavy).
    
    The bodies are initialised with random positions and velocities. 
    Each episode is rolled out for a specified number of steps and splited into a specified number of chunks.
    For example, batch_size = 5, length=100, chunks=10 would return 50 episodes of length 10 steps.
    
    Args:
        batch_size (int): The number of episodes to rollout in parallel.
        length (int): The length of the rollouts.
        chunks (int): The number of chunks to split the episodes.
        steps (int): The sampling frequency, e.g. 5 means sampling every 5th state in the sequence. 
        interventions(list[list[int]]): A list of lists specifying the interventions to apply. 
            For example, [[1], [2,3]] returns sequences from two environments, one with intervention 1 applied, 
            and the other with intervtnion 2 and 3 applied.
    
    Returns:
        The image and action sequences. Each sequence is of the shape (length/(chunks*steps), batch_size*chunks, len(interventions), image/action_dim).
    """
    data = get_dataset(batch_size, length, chunks, interventions)

    # Randomly sample the order in which the objects are renedered.
    # Note that the order is constant throughout each episode.
    orders = [np.random.permutation(4) for i in range(len(data[0][0]))]
    subsample_id = np.arange(0, len(data[0]), steps)
    im_array = [
        [
            [render(state, order) for state in inter]
            for (inter, order) in zip(batch, orders)
        ]
        for batch in data[0][subsample_id]
    ]
    np_images = (
        np.array(
            [
                np.array(
                    [np.array(list(map(np.asarray, image))) for image in trajectory]
                )
                for trajectory in im_array
            ]
        )
        / 255
    )
    return (np_images, data[1][subsample_id])
