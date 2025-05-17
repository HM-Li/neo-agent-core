import re


def str_starts_withs(s: str, starts: list[str]) -> bool:
    """
    s: string
    starts: list of strings to check if s starts with any of them
    """
    pattern = r"^" + "|".join(starts)
    if re.match(pattern, s):
        return True
    return False


import datetime


def get_current_utc_timestamp() -> str:
    """
    Get the current UTC timestamp in ISO 8601 format.
    """
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
