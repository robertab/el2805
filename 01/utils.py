#!/usr/bin/python3

def north(state):
    """
    Returns the new state by taking one step upwards

    usage:
    >>> north((1, 0))
    (0, 0)
    >>> north((3, 4))
    (2, 4)
    """
    return (state[0] - 1, state[1])

def south(state):
    """
    Returns the new state by taking one step upwards

    usage:
    >>> pos = (0, 0)
    >>> south(pos)
    (1, 0)

    """
    return (state[0] + 1, state[1])

def left(state):
    """
    Returns the new state by taking one step upwards

    usage:
    >>> pos = (4, 3)
    >>> left(pos)
    (4, 2)

    """
    return (state[0], state[1] - 1)

def right(state):
    """
    Returns the new state by taking one step upwards

    usage:
    >>> pos = (5, 2)
    >>> right(pos)
    (5, 3)
    """
    return (state[0], state[1] + 1)
