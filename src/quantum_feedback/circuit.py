"""Circuit validation, parameterized templates and parameter binding."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from .errors import CircuitValidationError

GATE_OPERATIONS = frozenset({"H", "X", "Z", "RZ", "CX"})
SINGLE_QUBIT_GATES = frozenset({"H", "X", "Z", "RZ"})


@dataclass(frozen=True)
class Operation:
    name: str
    qubit: int | None = None
    control: int | None = None
    target: int | None = None
    clbit: int | None = None
    angle: float | None = None
    probability: float | None = None
    condition_clbit: int | None = None
    condition_value: int | None = None


@dataclass(frozen=True)
class AngleExpression:
    """RZ angle of the form scale * parameter + offset."""

    parameter: str
    scale: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class ParameterizedOperation:
    name: str
    qubit: int | None = None
    control: int | None = None
    target: int | None = None
    clbit: int | None = None
    angle: "float | AngleExpression | None" = None
    probability: float | None = None
    condition_clbit: int | None = None
    condition_value: int | None = None


def validate_circuit(
    num_qubits: Any, num_clbits: Any, operations: Any
) -> list[Operation]:
    """Validate and bind a constant circuit without retaining caller data."""
    normalized = validate_template(num_qubits, num_clbits, operations)
    for position, operation in enumerate(normalized):
        if isinstance(operation.angle, AngleExpression):
            raise CircuitValidationError(
                f"operation {position}: parameterized RZ requires bind_parameters"
            )
    return [_constant_operation(operation) for operation in normalized]


def validate_template(
    num_qubits: Any, num_clbits: Any, operations: Any
) -> list[ParameterizedOperation]:
    """Validate a template that may contain named RZ angle parameters."""
    if not _is_plain_int(num_qubits) or not 1 <= num_qubits <= 6:
        raise CircuitValidationError("num_qubits must be an integer from 1 to 6")
    if not _is_plain_int(num_clbits) or not 1 <= num_clbits <= 6:
        raise CircuitValidationError("num_clbits must be an integer from 1 to 6")
    if not isinstance(operations, (list, tuple)):
        raise CircuitValidationError("operations must be a list or tuple")

    normalized: list[ParameterizedOperation] = []
    for position, raw_operation in enumerate(operations):
        normalized.append(_validate_operation(raw_operation, position, num_qubits, num_clbits))
    return normalized


def template_parameters(
    operations: "list[ParameterizedOperation] | tuple",
) -> tuple[str, ...]:
    """Return the named parameters used by a validated template, sorted."""
    names = {
        operation.angle.parameter
        for operation in operations
        if isinstance(operation.angle, AngleExpression)
    }
    return tuple(sorted(names))


def validate_parameters(
    operations: "list[ParameterizedOperation] | tuple",
    parameter_values: Any,
) -> dict[str, float]:
    """Fully validate a parameter value mapping without mutating caller data."""
    required = template_parameters(operations)
    if not isinstance(parameter_values, Mapping):
        raise CircuitValidationError("parameter_values must be a mapping")
    missing = [name for name in required if name not in parameter_values]
    if missing:
        raise CircuitValidationError(f"missing parameter value(s): {missing}")
    unexpected = sorted(set(parameter_values) - set(required))
    if unexpected:
        raise CircuitValidationError(f"unexpected parameter value(s): {unexpected}")
    bound: dict[str, float] = {}
    for name in required:
        value = parameter_values[name]
        if not isinstance(value, (int, float, np.floating, np.integer)) or isinstance(value, bool):
            raise CircuitValidationError(f"parameter {name!r}: value must be numeric")
        value = float(value)
        if not isfinite(value):
            raise CircuitValidationError(f"parameter {name!r}: value must be finite")
        bound[name] = value
    return bound


def bind_parameters(
    operations: "list[ParameterizedOperation] | tuple",
    parameter_values: Mapping[str, float],
) -> list[Operation]:
    """Bind all named parameters, leaving template and caller mapping untouched."""
    bound = validate_parameters(operations, parameter_values)
    return [_bind_operation(operation, bound) for operation in operations]


def _bind_operation(
    operation: ParameterizedOperation, bound: Mapping[str, float]
) -> Operation:
    angle = operation.angle
    if isinstance(angle, AngleExpression):
        angle = angle.scale * bound[angle.parameter] + angle.offset
        if not isfinite(angle):
            raise CircuitValidationError("bound RZ angle must be finite")
    return _constant_operation(operation, angle)


def _constant_operation(
    operation: ParameterizedOperation,
    angle: "float | AngleExpression | None" = None,
) -> Operation:
    if angle is None:
        angle = operation.angle
    return Operation(
        name=operation.name,
        qubit=operation.qubit,
        control=operation.control,
        target=operation.target,
        clbit=operation.clbit,
        angle=None if angle is None or isinstance(angle, AngleExpression) else float(angle),
        probability=operation.probability,
        condition_clbit=operation.condition_clbit,
        condition_value=operation.condition_value,
    )


def _validate_operation(
    raw_operation: Any, position: int, num_qubits: int, num_clbits: int
) -> ParameterizedOperation:
    where = f"operation {position}"
    if not isinstance(raw_operation, Mapping):
        raise CircuitValidationError(f"{where}: operation must be a mapping")

    operation = dict(raw_operation)
    name = operation.get("op")
    if not isinstance(name, str):
        raise CircuitValidationError(f"{where}: missing or non-string 'op'")

    condition = _validate_condition(operation.get("condition"), position, num_clbits)
    allowed = GATE_OPERATIONS | {"MEASURE", "RESET", "PHASE_FLIP"}
    if name not in allowed:
        raise CircuitValidationError(f"{where}: unknown operation {name!r}")
    if condition is not None and name not in GATE_OPERATIONS:
        raise CircuitValidationError(f"{where}: only gates may be conditional")

    if name in SINGLE_QUBIT_GATES:
        qubit = _required_qubit(operation, "qubit", position, num_qubits)
        angle = None
        if name == "RZ":
            angle = _required_angle(operation, position)
        _reject_extra_fields(operation, position, {"op", "qubit", "angle", "condition"})
        return ParameterizedOperation(
            name, qubit=qubit, angle=angle,
            condition_clbit=condition[0] if condition else None,
            condition_value=condition[1] if condition else None,
        )

    if name == "CX":
        control = _required_qubit(operation, "control", position, num_qubits)
        target = _required_qubit(operation, "target", position, num_qubits)
        if control == target:
            raise CircuitValidationError(f"{where}: CX control and target must be different")
        _reject_extra_fields(operation, position, {"op", "control", "target", "condition"})
        return ParameterizedOperation(
            name, control=control, target=target,
            condition_clbit=condition[0] if condition else None,
            condition_value=condition[1] if condition else None,
        )

    if name == "MEASURE":
        qubit = _required_qubit(operation, "qubit", position, num_qubits)
        clbit = _required_clbit(operation, position, num_clbits)
        _reject_extra_fields(operation, position, {"op", "qubit", "clbit"})
        return ParameterizedOperation(name, qubit=qubit, clbit=clbit)

    if name == "RESET":
        qubit = _required_qubit(operation, "qubit", position, num_qubits)
        _reject_extra_fields(operation, position, {"op", "qubit"})
        return ParameterizedOperation(name, qubit=qubit)

    qubit = _required_qubit(operation, "qubit", position, num_qubits)
    probability = _required_probability(operation, position)
    _reject_extra_fields(operation, position, {"op", "qubit", "p"})
    return ParameterizedOperation(name, qubit=qubit, probability=probability)


def _validate_condition(condition: Any, position: int, num_clbits: int) -> "tuple[int, int] | None":
    if condition is None:
        return None
    if not isinstance(condition, (list, tuple)) or len(condition) != 2:
        raise CircuitValidationError(f"operation {position}: condition must be [clbit, value]")
    clbit, value = condition
    if not _is_plain_int(clbit) or not 0 <= clbit < num_clbits:
        raise CircuitValidationError(f"operation {position}: condition classical bit is out of range")
    if not _is_plain_int(value) or value not in (0, 1):
        raise CircuitValidationError(f"operation {position}: condition value must be 0 or 1")
    return clbit, value


def _required_qubit(operation: Mapping, field: str, position: int, num_qubits: int) -> int:
    value = operation.get(field)
    if not _is_plain_int(value) or not 0 <= value < num_qubits:
        raise CircuitValidationError(f"operation {position}: {field} is out of range")
    return value


def _required_clbit(operation: Mapping, position: int, num_clbits: int) -> int:
    value = operation.get("clbit")
    if not _is_plain_int(value) or not 0 <= value < num_clbits:
        raise CircuitValidationError(f"operation {position}: clbit is out of range")
    return value


def _required_angle(operation: Mapping, position: int) -> "float | AngleExpression":
    angle = operation.get("angle")
    if isinstance(angle, str):
        return _angle_from_mapping({"param": angle}, position)
    if isinstance(angle, Mapping):
        return _angle_from_mapping(angle, position)
    if not isinstance(angle, (int, float, np.floating, np.integer)) or isinstance(angle, bool):
        raise CircuitValidationError(f"operation {position}: RZ angle must be numeric or parameterized")
    angle = float(angle)
    if not isfinite(angle):
        raise CircuitValidationError(f"operation {position}: RZ angle must be finite")
    return angle


def _angle_from_mapping(angle: Mapping, position: int) -> AngleExpression:
    parameter = angle.get("param")
    if not isinstance(parameter, str) or not parameter:
        raise CircuitValidationError(f"operation {position}: angle 'param' must be a non-empty string")
    scale = angle.get("scale", 1.0)
    if not isinstance(scale, (int, float, np.floating, np.integer)) or isinstance(scale, bool):
        raise CircuitValidationError(f"operation {position}: angle 'scale' must be numeric")
    scale = float(scale)
    if not isfinite(scale) or scale == 0.0:
        raise CircuitValidationError(f"operation {position}: angle 'scale' must be finite and non-zero")
    offset = angle.get("offset", 0.0)
    if not isinstance(offset, (int, float, np.floating, np.integer)) or isinstance(offset, bool):
        raise CircuitValidationError(f"operation {position}: angle 'offset' must be numeric")
    offset = float(offset)
    if not isfinite(offset):
        raise CircuitValidationError(f"operation {position}: angle 'offset' must be finite")
    extra = set(angle) - {"param", "scale", "offset"}
    if extra:
        raise CircuitValidationError(f"operation {position}: unexpected angle field(s): {sorted(extra)}")
    return AngleExpression(parameter, scale, offset)


def _required_probability(operation: Mapping, position: int) -> float:
    probability = operation.get("p")
    if not isinstance(probability, (int, float, np.floating, np.integer)) or isinstance(probability, bool):
        raise CircuitValidationError(f"operation {position}: phase-flip probability must be numeric")
    probability = float(probability)
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise CircuitValidationError(f"operation {position}: phase-flip probability must be between 0 and 1")
    return probability


def _reject_extra_fields(operation: Mapping, position: int, allowed: set[str]) -> None:
    extra = set(operation) - allowed
    if extra:
        raise CircuitValidationError(f"operation {position}: unexpected field(s): {sorted(extra)}")


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
