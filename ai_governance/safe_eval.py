"""Safe expression evaluator for rollback conditions.

Evaluates expressions like `fraud_claim_escalation_rate < 0.60 and fraud_claim_count >= 10`
using AST walking. Only allows: names (variable lookups), numeric literals, boolean literals,
comparisons (<, >, <=, >=, ==, !=), boolean operators (and, or, not), and unary minus.

No attribute access, no subscripts, no calls, no comprehensions — blocks the
`().__class__.__bases__[0].__subclasses__()` introspection chain entirely.
"""

from __future__ import annotations

import ast
import operator
from typing import Any


_CMP_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

_BOOL_OPS = {
    ast.And: all,
    ast.Or: any,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}


class SafeExprError(ValueError):
    pass


def safe_eval(expression: str, context: dict[str, Any]) -> bool:
    """Evaluate a boolean expression safely against the given context.

    Raises SafeExprError on disallowed AST nodes or unknown variables.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeExprError(f"Invalid expression syntax: {exc}") from exc
    return bool(_eval_node(tree.body, context))


def _eval_node(node: ast.AST, ctx: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, ctx)

    if isinstance(node, ast.BoolOp):
        op_fn = _BOOL_OPS.get(type(node.op))
        if op_fn is None:
            raise SafeExprError(f"Disallowed boolean operator: {type(node.op).__name__}")
        return op_fn(_eval_node(v, ctx) for v in node.values)

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, ctx)
        for op, comparator in zip(node.ops, node.comparators):
            op_fn = _CMP_OPS.get(type(op))
            if op_fn is None:
                raise SafeExprError(
                    f"Disallowed comparison: {type(op).__name__}. "
                    f"Allowed: <, >, <=, >=, ==, !="
                )
            right = _eval_node(comparator, ctx)
            if not op_fn(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise SafeExprError(f"Disallowed unary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand, ctx))

    if isinstance(node, ast.BinOp):
        bin_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }
        op_fn = bin_ops.get(type(node.op))
        if op_fn is None:
            raise SafeExprError(f"Disallowed binary operator: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left, ctx), _eval_node(node.right, ctx))

    if isinstance(node, ast.Name):
        if node.id not in ctx:
            raise SafeExprError(f"Unknown variable: {node.id!r}")
        return ctx[node.id]

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise SafeExprError(f"Disallowed constant type: {type(node.value).__name__}")

    raise SafeExprError(
        f"Disallowed expression node: {type(node).__name__}. "
        f"Only comparisons, boolean operators, names, and numbers are allowed."
    )
