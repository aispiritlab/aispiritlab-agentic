from datetime import datetime


def get_current_time() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate, e.g. '2 + 2' or '(10 * 5) / 3'.
    """
    allowed_chars = set("0123456789+-*/().% ")
    if not all(c in allowed_chars for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def roll_dice(notation: str) -> str:
    """Roll dice using standard notation like '2d6' (2 six-sided dice).

    Args:
        notation: Dice notation in NdM format, e.g. '1d20', '3d6', '2d8'.
    """
    import random

    notation = notation.strip().lower()
    if "d" not in notation:
        return "Error: use NdM format, e.g. '2d6'."
    try:
        parts = notation.split("d")
        count = int(parts[0]) if parts[0] else 1
        sides = int(parts[1])
    except (ValueError, IndexError):
        return "Error: invalid dice notation. Use NdM format, e.g. '2d6'."

    if count < 1 or count > 100 or sides < 2 or sides > 1000:
        return "Error: count must be 1-100, sides must be 2-1000."

    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    return f"Rolls: {rolls}, Total: {total}"
