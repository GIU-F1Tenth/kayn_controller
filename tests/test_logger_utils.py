import sys, os
from unittest.mock import MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from kayn_controller.logger_utils import KAYNLogger, LogLevel


def _make_logger(component="Comp", level="normal"):
    node = MagicMock()
    ros_log = MagicMock()
    node.get_logger.return_value = ros_log
    return KAYNLogger(node, component, level), ros_log


def test_info_includes_component_tag():
    kl, ros_log = _make_logger("MyComp")
    kl.info("hello")
    ros_log.info.assert_called_once()
    msg = ros_log.info.call_args[0][0]
    assert "[MyComp]" in msg
    assert "hello" in msg


def test_warn_includes_warning_prefix():
    kl, ros_log = _make_logger()
    kl.warn("something wrong")
    ros_log.warn.assert_called_once()
    msg = ros_log.warn.call_args[0][0]
    assert "WARNING" in msg or "⚠" in msg
    assert "something wrong" in msg


def test_error_includes_exception_text():
    kl, ros_log = _make_logger()
    kl.error("bad thing", exception=ValueError("boom"))
    ros_log.error.assert_called_once()
    msg = ros_log.error.call_args[0][0]
    assert "boom" in msg


def test_critical_uses_error_channel():
    kl, ros_log = _make_logger()
    kl.critical("system failure")
    ros_log.error.assert_called_once()


def test_startup_always_logs():
    kl, ros_log = _make_logger(level="minimal")
    kl.startup("node online")
    ros_log.info.assert_called_once()
    msg = ros_log.info.call_args[0][0]
    assert "node online" in msg


def test_debug_suppressed_at_normal_level():
    kl, ros_log = _make_logger(level="normal")
    kl.debug("verbose detail")
    ros_log.info.assert_not_called()


def test_debug_visible_at_debug_level():
    kl, ros_log = _make_logger(level="debug")
    kl.debug("verbose detail")
    ros_log.info.assert_called_once()


def test_event_includes_name_and_details():
    kl, ros_log = _make_logger()
    kl.event("lap_start", details="lap=3")
    ros_log.info.assert_called_once()
    msg = ros_log.info.call_args[0][0]
    assert "lap_start" in msg
    assert "lap=3" in msg


def test_invalid_log_level_defaults_to_normal():
    kl, ros_log = _make_logger(level="nonsense")
    assert kl.log_level == LogLevel.NORMAL
