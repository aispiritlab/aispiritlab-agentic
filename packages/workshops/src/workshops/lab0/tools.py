from __future__ import annotations

import ast
from datetime import UTC, datetime
import operator

#: Operators the calculator accepts. Anything else is rejected at parse time.
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Guards against results that are cheap to write and expensive to compute.
_MAX_ABS_OPERAND = 10**15


def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _evaluate_node(node: ast.expr) -> float:
    """Evaluate one arithmetic node.

    A restricted AST walk rather than ``eval``: a character allowlist still lets
    ``9**9**9`` through, which hangs the process computing a number nobody asked
    for. Exponentiation is simply not part of the grammar here.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        if abs(node.value) > _MAX_ABS_OPERAND:
            raise ValueError("number is too large")
        return node.value

    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPERATORS.get(type(node.op))
        if unary is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return unary(_evaluate_node(node.operand))

    if isinstance(node, ast.BinOp):
        binary = _BINARY_OPERATORS.get(type(node.op))
        if binary is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return binary(_evaluate_node(node.left), _evaluate_node(node.right))

    raise ValueError(f"unsupported expression: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Supports + - * / // % and parentheses on plain numbers.

    Args:
        expression: A mathematical expression to evaluate, e.g. '2 + 2' or '(10 * 5) / 3'.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return "Error: that is not a valid arithmetic expression."

    try:
        return str(_evaluate_node(tree.body))
    except ZeroDivisionError:
        return "Error: division by zero."
    except ValueError as error:
        return f"Error: {error}"


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
    except ValueError, IndexError:
        return "Error: invalid dice notation. Use NdM format, e.g. '2d6'."

    if count < 1 or count > 100 or sides < 2 or sides > 1000:
        return "Error: count must be 1-100, sides must be 2-1000."

    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls)
    return f"Rolls: {rolls}, Total: {total}"
