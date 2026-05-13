from enum import Enum
from typing import Optional
from rclpy.node import Node


class LogLevel(Enum):
    MINIMAL = "minimal"
    NORMAL  = "normal"
    DEBUG   = "debug"
    VERBOSE = "verbose"


_LEVEL_ORDER = {
    LogLevel.MINIMAL: 0,
    LogLevel.NORMAL:  1,
    LogLevel.DEBUG:   2,
    LogLevel.VERBOSE: 3,
}


class KAYNLogger:
    """Structured logging wrapper for KAYN controller components."""

    def __init__(self, node: Node, component_name: str, log_level: str = "normal"):
        self.node = node
        self.component = component_name
        self._set_log_level(log_level)

    def _set_log_level(self, level: str):
        try:
            self.log_level = LogLevel(level.lower())
        except ValueError:
            self.node.get_logger().warn(
                f"[{self.component}] Invalid log level '{level}'. Using 'normal'."
            )
            self.log_level = LogLevel.NORMAL

    def _fmt(self, message: str, prefix: str = "") -> str:
        tag = f"[{self.component}]"
        return f"{tag} {prefix} {message}" if prefix else f"{tag} {message}"

    def _should_log(self, required: LogLevel) -> bool:
        return _LEVEL_ORDER[self.log_level] >= _LEVEL_ORDER[required]

    # ── always-on ────────────────────────────────────────────────────────────

    def error(self, message: str, exception: Optional[Exception] = None):
        text = self._fmt(message, "❌ ERROR:")
        if exception:
            text += f" | Exception: {exception}"
        self.node.get_logger().error(text)

    def critical(self, message: str):
        self.node.get_logger().error(self._fmt(message, "🚨 CRITICAL:"))

    def startup(self, message: str):
        self.node.get_logger().info(self._fmt(message, "STARTUP:"))

    def shutdown(self, message: str):
        self.node.get_logger().info(self._fmt(message, "🛑 SHUTDOWN:"))

    # ── level-gated ──────────────────────────────────────────────────────────

    def warn(self, message: str, level: LogLevel = LogLevel.NORMAL):
        if self._should_log(level):
            self.node.get_logger().warn(self._fmt(message, "⚠️  WARNING:"))

    def info(self, message: str, level: LogLevel = LogLevel.NORMAL):
        if self._should_log(level):
            self.node.get_logger().info(self._fmt(message))

    def success(self, message: str, level: LogLevel = LogLevel.NORMAL):
        if self._should_log(level):
            self.node.get_logger().info(self._fmt(message, "✓"))

    def debug(self, message: str):
        if self._should_log(LogLevel.DEBUG):
            self.node.get_logger().info(self._fmt(message, "🔍 DEBUG:"))

    def verbose(self, message: str):
        if self._should_log(LogLevel.VERBOSE):
            self.node.get_logger().info(self._fmt(message, "📝 VERBOSE:"))

    # ── specialised ──────────────────────────────────────────────────────────

    def event(self, event_name: str, details: str = "", level: LogLevel = LogLevel.NORMAL):
        if self._should_log(level):
            msg = f"📍 EVENT: {event_name}"
            if details:
                msg += f" | {details}"
            self.node.get_logger().info(self._fmt(msg))

    def metric(self, name: str, value, unit: str = "", level: LogLevel = LogLevel.NORMAL):
        if self._should_log(level):
            val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            msg = f"-> {name}: {val_str}"
            if unit:
                msg += f" {unit}"
            self.node.get_logger().info(self._fmt(msg))

    def status(self, status: str, level: LogLevel = LogLevel.NORMAL):
        if self._should_log(level):
            self.node.get_logger().info(self._fmt(status, "🔄 STATUS:"))

    def config(self, parameter: str, value, level: LogLevel = LogLevel.DEBUG):
        if self._should_log(level):
            self.node.get_logger().info(self._fmt(f"⚙️  CONFIG: {parameter} = {value}"))
