"""Strict Phase 8 model responses; no benchmark labels enter these contracts."""
import ast
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

EXPRESSION_MAX_CHARS = 200
EXPRESSION_ALLOWED_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
EXPRESSION_ALLOWED_UNARY = (ast.UAdd, ast.USub)


def validate_expression_syntax(expression):
    """Whitelist arithmetic syntax only: no names, calls, attributes or containers."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError('Expression must be a nonempty string.')
    if len(expression) > EXPRESSION_MAX_CHARS:
        raise ValueError(f'Expression exceeds {EXPRESSION_MAX_CHARS} characters.')
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as exc:
        raise ValueError(f'Expression is not valid arithmetic syntax: {exc}') from exc
    constants = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            if type(node.op) not in EXPRESSION_ALLOWED_OPS:
                raise ValueError('Only +, -, *, / and ** are allowed.')
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in EXPRESSION_ALLOWED_UNARY:
                raise ValueError('Only unary + and - are allowed.')
        elif isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ValueError('Only numeric constants are allowed.')
            constants += 1
        elif isinstance(node, (ast.Expression, *EXPRESSION_ALLOWED_OPS, *EXPRESSION_ALLOWED_UNARY)):
            continue
        else:
            raise ValueError(f'Disallowed syntax element: {type(node).__name__}.')
    if constants < 1:
        raise ValueError('Expression must contain at least one number.')
    if constants > 30:
        raise ValueError('Expression contains too many constants.')
    return expression


class Response(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Answer(Response):
    action: Literal['answer']
    answer: str = Field(min_length=1)
    citations: list[str] = Field(min_length=1)


class Abstain(Response):
    action: Literal['abstain']
    # A bare abstain action is a valid refusal, not a pipeline failure.
    reason: str = Field(default='Insufficient evidence', min_length=1)


class Calculation(Response):
    action: Literal['calculate']
    # Two forms: legacy operation+operands, or one whitelisted arithmetic expression.
    operation: Literal['add', 'subtract', 'multiply', 'divide', 'mean', 'percentage_change'] | None = None
    operands: list[int | float] | None = None
    expression: str | None = None
    citations: list[str] = Field(min_length=1)
    units: str = Field(min_length=1)

    @model_validator(mode='after')
    def exactly_one_form(self):
        has_legacy = self.operation is not None or self.operands is not None
        has_expression = self.expression is not None
        if has_legacy == has_expression:
            raise ValueError('Provide exactly one of expression or operation+operands.')
        if has_expression:
            validate_expression_syntax(self.expression)
            return self
        if self.operation is None or self.operands is None:
            raise ValueError('operation and operands must be provided together.')
        if not 1 <= len(self.operands) <= 20:
            raise ValueError('Provide 1 to 20 operands.')
        if self.operation == 'subtract' and len(self.operands) < 2:
            raise ValueError('Subtraction requires at least two operands.')
        if self.operation in ('multiply', 'divide', 'percentage_change') and len(self.operands) != 2:
            raise ValueError('This operation requires exactly two operands.')
        return self


class Rewrite(Response):
    query: str = Field(min_length=1)

    @model_validator(mode='after')
    def nonempty_query(self):
        if not self.query.strip():
            raise ValueError('A retrieval retry needs a nonempty query.')
        return self


class Assessment(Response):
    sufficient: bool
    missing_evidence: str
    query: str

    @model_validator(mode='after')
    def retry_query(self):
        if not self.sufficient and not self.query.strip():
            raise ValueError('A retrieval retry needs a nonempty query.')
        return self


FINAL_RESPONSE = TypeAdapter(Answer | Abstain)
ANSWER_OR_CALCULATION = TypeAdapter(Answer | Abstain | Calculation)
REWRITE_RESPONSE = TypeAdapter(Rewrite)
ASSESS_RESPONSE = TypeAdapter(Assessment)
