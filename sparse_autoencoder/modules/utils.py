def is_power_of_2(n: int) -> bool:
    """Check if `n` is a power of 2."""
    return n and not (n & (n - 1))
