from data.multi_body import init_body, run_simulation, pd_control, render
import numpy as np


def get_states(
    batch_size: int,
    length: int = 1000,
    chunks: int = 20,
    interventions: list[list[int]] = [[]],
) -> tuple[np.ndarray, np.ndarray]:
    """ Returns the state and action sequences.

    The bodies are initialised with random positions and velocities.
    Each episode is rolled out for a specified number of steps and splited into a
    specified number of chunks. For example, batch_size = 5, length=100, chunks=10
    would return 50 episodes of length 10 steps.

    Args:
        batch_size (int): The number of episodes to rollout in parallel.
        length (int): The length of the rollouts.
        chunks (int): The number of chunks to split the episodes.
        interventions(list[list[int]]): A list of lists specifying the interventions to
            apply. For example, [[1], [2,3]] returns sequences from two environments,
            one with intervention 1 applied, and the other with intervtnion 2 and 3
            applied.

    Returns:
        The state and action sequences. Each sequence is of the shape
            (length/chunks, batch_size*chunks, len(interventions), state/action_dim).

    """
    state_data = []
    action_data = []
    for inter in interventions:
        if 3 in inter:
            b1 = init_body(batch_size, 2)
        elif 6 in inter:
            b1 = init_body(batch_size, 0.5)
        else:
            b1 = init_body(batch_size)

        if 4 in inter:
            b2 = init_body(batch_size, 2)
        elif 7 in inter:
            b2 = init_body(batch_size, 0.5)
        else:
            b2 = init_body(batch_size)

        if 5 in inter:
            b3 = init_body(batch_size, 2)
        elif 8 in inter:
            b3 = init_body(batch_size, 0.5)
        else:
            b3 = init_body(batch_size)

        a = init_body(batch_size)
        state_log = []
        action_log = []
        for i in range(length):
            action = pd_control(a, np.array([2.0, 2.0]))
            b1, b2, b3, a = run_simulation(b1, b2, b3, a, action, interventions=inter)
            state_log.append(
                np.concatenate(
                    [b1.position, b2.position, b3.position, a.position], axis=-1,
                )
            )
            action_log.append(action)
        state_log, action_log = (
            np.stack(state_log),
            np.stack(action_log),
        )
        state_data.append(np.concatenate(np.split(state_log, chunks, axis=0), axis=1))
        action_data.append(np.concatenate(np.split(action_log, chunks, axis=0), axis=1))
    return np.stack(state_data, 2), np.stack(action_data, 2)


def get_images(
    batch_size: int,
    length: int = 1000,
    chunks: int = 20,
    steps: int = 5,
    interventions: list[int] = [[]],
) -> tuple[np.ndarray, np.ndarray]:
    """ Generates an image dataset (Note that this is memory heavy).

    The bodies are initialised with random positions and velocities.
    Each episode is rolled out for a specified number of steps and splited into a
    specified number of chunks. For example, batch_size = 5, length=100, chunks=10
    would return 50 episodes of length 10 steps.

    Args:
        batch_size (int): The number of episodes to rollout in parallel.
        length (int): The length of the rollouts.
        chunks (int): The number of chunks to split the episodes.
        interventions(list[list[int]]): A list of lists specifying the interventions to
            apply. For example, [[1], [2,3]] returns sequences from two environments,
            one with intervention 1 applied, and the other with intervtnion 2 and 3
            applied.

    Returns:
        The image and action sequences. Each sequence is of the shape
        (length/(chunks*steps), batch_size*chunks, len(interventions), image/action_dim)
    """
    data = get_states(batch_size, length, chunks, interventions)

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
    np_images = np.array(
        [
            np.array([np.array(list(map(np.asarray, image))) for image in trajectory])
            for trajectory in im_array
        ]
    )
    return (np_images, data[1][subsample_id])


def get_flattened_images(
    batch_size: int,
    length: int = 1000,
    chunks: int = 20,
    steps: int = 5,
    interventions: list[int] = [[]],
) -> tuple[np.ndarray, np.ndarray]:
    data = get_images(batch_size, length, chunks, steps, interventions)
    images = data[0].reshape(1, -1, 128, 128, 3)
    actions = data[1].reshape(1, -1, 2)
    p = np.random.permutation(images.shape[1])
    return (images[:, p], actions[:, p])
