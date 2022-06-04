from re import L
import numpy as np
from PIL import Image, ImageDraw
from dataclasses import dataclass
from typing import Union, Tuple, Callable


@dataclass
class Body:
    """ This is a dataclass containing the states of an object."""

    position: np.ndarray
    velocity: np.ndarray
    mass: float
    force: np.ndarray


def init_body(batch_size: int, mass: float = 1.0) -> Body:
    """ Returns an initialised body object.
    
    Args:
        batch_size (int): The size of a batch.
        mass (float): The mass of the object.
    
    Returns:
        A body object with normally distributed position and velocity.
    """
    position = np.random.randn(batch_size, 2)
    velocity = np.random.randn(batch_size, 2) * 2
    mass = np.ones((batch_size, 1)) * mass
    force = np.zeros_like(position)
    return Body(position, velocity, mass, force)


def step_body(state: Body, dt: float, restrict: Union[str, None] = None) -> Body:
    """ Updates the state of a body according to the velocity and acceleration.

    Args:
        state (Body): The body object to be updated.
        dt (float): The size of the timestep.
        restrict (Union[str, None]): The dimension of movement to be restricted,
            e.g. 'x' means that the object does not move in the x direction.
    
    Returns:
        The updated body object.
    """
    position = state.position + state.velocity * dt
    acceleration = state.force / state.mass
    velocity = state.velocity + acceleration * dt
    mass = state.mass

    # Restrict the velocity of the object.
    if restrict == "x":
        velocity[..., 0] = 0
    if restrict == "y":
        velocity[..., 1] = 0

    # Bounce from the boundaries if object is out of bounds.
    large_x = position[..., 0] > 3
    position[large_x, 0] = 6 - position[large_x, 0]
    velocity[large_x, 0] = -velocity[large_x, 0]

    small_x = position[..., 0] < -3
    position[small_x, 0] = -6 - position[small_x, 0]
    velocity[small_x, 0] = -velocity[small_x, 0]

    large_y = position[..., 1] > 3
    position[large_y, 1] = 6 - position[large_y, 1]
    velocity[large_y, 1] = -velocity[large_y, 1]

    small_y = position[..., 1] < -3
    position[small_y, 1] = -6 - position[small_y, 1]
    velocity[small_y, 1] = -velocity[small_y, 1]

    force = np.zeros_like(position)
    return Body(position, velocity, mass, force)
    # return state


def apply_force(state: Body, force: np.ndarray) -> Body:
    """ Updates the force on an object."""
    state.force = state.force + force
    return state


def apply_action(state: Body, action: np.ndarray) -> Body:
    """ Applies a thrust on an object."""
    # The action is soft-clipped at +- 3.
    thrust = 3.0 * np.tanh(action / 3)
    return apply_force(state, thrust)


def spring_force(b1: Body, b2: Body, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the forces between two ojects linked with spring.
    
    Args:
        b1 (Body): The first body.
        b2 (Body): The second body.
        k (float): The spring constant.

    Returns:
        The forces on both objects.
    """
    f1 = k * (b2.position - b1.position)
    return f1, -f1


def gravitational_force(b1: Body, b2: Body, k: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the forces between two objects linked gravitationally.
    
    Args:
        b1 (Body): The first body.
        b2 (Body): The second body.
        k (float): The gravitational constant.

    Returns:
        The forces on both objects.
    """
    f1 = (
        k
        * (b2.position - b1.position)
        * b1.mass
        * b2.mass
        / np.expand_dims(((b2.position - b1.position) ** 2).sum(axis=-1), -1)
    )
    return f1, -f1


def pd_control(a: Body, goal: np.ndarray = np.array([2.0, 2.0])) -> np.ndarray:
    """ This is a controller that drives the agent to a specified goal.
    
    Args:
        a (Body): The state of the agent Body.
        goal (np.ndarray): The goal coordinates.

    Returns:
        An action vector that drives the agent to the goal.
    """
    prop = goal - a.position
    diff = -a.velocity
    return diff + prop


def step_simulation(
    b1: Body,
    b2: Body,
    b3: Body,
    a: Body,
    action: np.ndarray,
    interventions: list[int] = [],
    dt: float = 0.001,
) -> Tuple[(Body,) * 4]:
    """ Computes the forces between the obejcts and update the states.

    Args: 
        b1 (Body): Body number 1.
        b2 (Body): Body number 2.
        b3 (Body): Body number 3.
        a (Body): The agent.
        action (np.ndarray): The action vector to be applied to the agent.
        interventions (list[int]): A list containing the interventions applied to the environment. See table 1.
        dt (float): The size of the timestep.

    Returns:
        A tuple of the four bodies after an update step.
    """
    a = apply_action(a, action)
    force_a, _ = gravitational_force(a, b1, 3)
    a = apply_force(a, force_a)
    force_a, _ = gravitational_force(a, b3, -3)
    a = apply_force(a, force_a)
    if 1 in interventions:
        force1, force2 = 0, 0
    else:
        if 9 in interventions:
            force1, force2 = spring_force(b1, b2, 4)
        else:
            force1, force2 = spring_force(b1, b2, 1)
    b1 = apply_force(b1, force1)
    b2 = apply_force(b2, force2)
    if 2 in interventions:
        force2, force3 = 0, 0
    else:
        if 10 in interventions:
            force2, force3 = spring_force(b2, b3, 4)
        else:
            force2, force3 = spring_force(b2, b3, 1)
    b2 = apply_force(b2, force2)
    b3 = apply_force(b3, force3)

    # step bodies
    if 11 in interventions:
        b1 = step_body(b1, dt, "x")
    elif 12 in interventions:
        b1 = step_body(b1, dt, "y")
    else:
        b1 = step_body(b1, dt)

    if 13 in interventions:
        b2 = step_body(b2, dt, "x")
    elif 14 in interventions:
        b2 = step_body(b2, dt, "y")
    else:
        b2 = step_body(b2, dt)

    if 15 in interventions:
        b3 = step_body(b3, dt, "x")
    elif 16 in interventions:
        b3 = step_body(b3, dt, "y")
    else:
        b3 = step_body(b3, dt)

    if 17 in interventions:
        a = step_body(a, dt, "x")
    elif 18 in interventions:
        a = step_body(a, dt, "y")
    else:
        a = step_body(a, dt)
    return b1, b2, b3, a


def run_simulation(
    b1: Body,
    b2: Body,
    b3: Body,
    a: Body,
    action: np.ndarray,
    step_fn: Callable = step_simulation,
    dt: float = 0.002,
    steps: int = 20,
    interventions: list[int] = [],
):
    """ Perform a number of simulation steps.
    
    Args: 
        b1 (Body): Body number 1.
        b2 (Body): Body number 2.
        b3 (Body): Body number 3.
        a (Body): The agent.
        action (np.ndarray): The action vector to be applied to the agent.
        step_fn (Callable): A function that performs a step in the simulation.
        dt (float): The size of the timestep.
        steps (int): The number of steps to take.
        interventions (list[int]): A list containing the interventions applied to the environment. See table 1.

    Returns:
        A tuple of the four bodies after the update steps.
        """
    for i in range(steps):
        b1, b2, b3, a = step_fn(
            b1, b2, b3, a, action, interventions=interventions, dt=dt
        )
    return b1, b2, b3, a


def get_dataset(
    batch_size: int,
    length: int = 1000,
    chunks: int = 20,
    interventions: list[list[int]] = [[]],
) -> Tuple[np.ndarray, np.ndarray]:
    """ Returns the state and action sequences. 
    
    The bodies are initialised with random positions and velocities. 
    Each episode is rolled out for a specified number of steps and splited into a specified number of chunks.
    For example, batch_size = 5, length=100, chunks=10 would return 50 episodes of length 10 steps.
    
    Args:
        batch_size (int): The number of episodes to rollout in parallel.
        length (int): The length of the rollouts.
        chunks (int): The number of chunks to split the episodes.
        interventions(list[list[int]]): A list of lists specifying the interventions to apply. 
            For example, [[1], [2,3]] returns sequences from two environments, one with intervention 1 applied, 
            and the other with intervtnion 2 and 3 applied.

    Returns:
        The state and action sequences. Each sequence is of the shape (length/chunks, batch_size*chunks, len(interventions), state/action_dim).

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


def render(state: np.ndarray, order: tuple[(int,) * 4]) -> Image:
    """ Renders the state.
    
    Args:
        state (np.ndarray): The state vector (x1, y1, x2, y2, x3, y3, x_agent, y_agent).
        order (tuple[int]): The order in which the objects are rendered, 
            i.e. [1,2,3,4] means body1 will be occluded by all other objects if overlapped.
    
    Returns:
        The rendered image.
    """
    grid_size = 128
    im = Image.new("RGB", (grid_size, grid_size), (128, 128, 128))
    draw_ball1 = ImageDraw.Draw(im)
    draw_ball2 = ImageDraw.Draw(im)
    draw_ball3 = ImageDraw.Draw(im)
    draw_agent = ImageDraw.Draw(im)
    size_1 = 20
    size_2 = 20
    size_3 = 20
    size_a = 20
    to_coor = lambda x: x / 3 * (103 / 2) + (103 / 2)

    for i in order:
        if i == 0:
            draw_ball1.ellipse(
                (
                    int(to_coor(state[0])),
                    int(to_coor(state[1])),
                    int(to_coor(state[0]) + size_1),
                    int(to_coor(state[1]) + size_1),
                ),
                fill=(255, 204, 51),
            )
        elif i == 1:
            draw_ball2.ellipse(
                (
                    int(to_coor(state[2])),
                    int(to_coor(state[3])),
                    int(to_coor(state[2]) + size_2),
                    int(to_coor(state[3]) + size_2),
                ),
                fill=(51, 255, 102),
            )
        elif i == 2:
            draw_ball3.ellipse(
                (
                    int(to_coor(state[4])),
                    int(to_coor(state[5])),
                    int(to_coor(state[4]) + size_3),
                    int(to_coor(state[5]) + size_3),
                ),
                fill=(51, 102, 255),
            )
        else:
            draw_agent.ellipse(
                (
                    int(to_coor(state[6])),
                    int(to_coor(state[7])),
                    int(to_coor(state[6]) + size_a),
                    int(to_coor(state[7]) + size_a),
                ),
                fill=(255, 51, 204),
            )
    return im
