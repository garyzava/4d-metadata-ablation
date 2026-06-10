"""Value serialization for the gold-result cache.

`serialize_value` coerces a DB value returned by SQLAlchemy into a
JSON-serializable scalar so the BIRD-EX gold cache (precompute_gold.py)
round-trips. It only needs to JSON-clean the values; the set-equality
comparison itself is normalized separately in eval/runner.py
(`_bird_ex_match._normalize_row`, which casts to float/str/None).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Any


def serialize_value(v: Any) -> Any:
    """Decimal -> float, date/time -> ISO string, bytes -> utf-8; JSON-native passthrough."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).decode("utf-8", "replace")
    return str(v)
