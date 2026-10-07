"""Exception hierarchy shared by every subsystem."""

from __future__ import annotations


class ControlError(Exception):
    """Base class for every error raised by the control service."""

    code = "control_error"

    def as_dict(self) -> dict[str, str]:
        return {"error": str(self), "code": self.code}


class ValidationError(ControlError):
    """An entity or payload failed a structural check."""

    code = "validation_error"


class NotFoundError(ControlError):
    """A requested entity does not exist."""

    code = "not_found"


class ConflictError(ControlError):
    """The request contradicts the committed state."""

    code = "conflict"


class OrderingError(ControlError):
    """A step was attempted out of the order the plant allows."""

    code = "ordering_error"


class StaleError(ControlError):
    """A generation, watermark or schedule reference is out of date."""

    code = "stale_reference"


class QuotaError(ControlError):
    """A quota or capacity bound rejected the request."""

    code = "quota_exceeded"


class BatchRemapError(ConflictError):
    """A remap batch was rejected; every failure names the item that caused it."""

    code = "batch_remap_rejected"

    def __init__(self, failures: list[dict[str, object]]) -> None:
        self.failures = [dict(failure) for failure in failures]
        bags = ", ".join(str(failure.get("bag_id", "?")) for failure in self.failures)
        super().__init__(
            f"remap batch rejected: {len(self.failures)} item(s) failed ({bags})"
        )

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = dict(super().as_dict())
        payload["failures"] = [dict(failure) for failure in self.failures]
        return payload
