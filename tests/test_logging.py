import json
import logging

from utils.logging_setup import JsonFormatter


def test_json_formatter_writes_valid_json() -> None:
    record = logging.makeLogRecord(
        {"name": "t", "levelno": logging.INFO, "levelname": "INFO",
         "msg": "hello", "data": {"seed": 7}}
    )
    parsed = json.loads(JsonFormatter().format(record))
    assert parsed["message"] == "hello"
    assert parsed["seed"] == 7
    assert parsed["level"] == "INFO"