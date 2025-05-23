from datetime import timedelta
import datetime
import re
import sys

def parse_duration(duration_str: str) -> timedelta:
    """
    Parses a duration string like '1H', '30min', '2D', '1h30m', '2days 5hours', '10s'
    into a datetime.timedelta object.

    Supports the following units:
        - H, h: hours
        - min, m: minutes
        - D, d, days: days
        - S, s: seconds
        - W, w, weeks: weeks

    Args:
        duration_str (str): The duration string to parse.

    Returns:
        datetime.timedelta: A timedelta object representing the duration.
                            Returns timedelta(0) for "0", None, or empty string.

    Raises:
        ValueError: If the duration string is invalid and not an explicit zero value.
    """
    if duration_str == "0" or duration_str is None or duration_str == "":
        return timedelta(seconds=0)

    # Regex to find all number-unit pairs (e.g., "10s", "2h", "30m")
    # Allows for optional space between number and unit.
    pattern = r'(\d+)\s*([HhmindaysSwW]+)' 
    matches = re.findall(pattern, duration_str)

    if not matches:
        # If the string is not "0" and no matches found, it's an invalid format.
        # However, if it's just a number, treat it as seconds.
        try:
            # Check if the entire string is just a number (integer or float)
            # This handles cases like "10" (meaning 10 seconds) if not captured by regex.
            # Only do this if the regex found nothing, to avoid double-counting.
            num_seconds = float(duration_str)
            if num_seconds >= 0: # Ensure it's not negative if parsed this way
                 return timedelta(seconds=num_seconds)
            else:
                raise ValueError(f"Invalid duration format: '{duration_str}'. Negative bare number.")
        except ValueError: # Not a simple number, and regex failed
            raise ValueError(f"Invalid duration format: '{duration_str}'. No valid units found.")

    total_seconds = 0
    for value_str, unit in matches:
        value = int(value_str)
        if unit.lower() in ('h'):
            total_seconds += value * 3600
        elif unit.lower() in ('min', 'm'):
            total_seconds += value * 60
        elif unit.lower() in ('d', 'days'):
            total_seconds += value * 86400
        elif unit.lower() in ('s'):
            total_seconds += value
        elif unit.lower() in ('w', 'weeks'):
            total_seconds += value * 604800
        else:
            # This case should ideally not be reached if regex is comprehensive for supported units
            raise ValueError(f"Unknown time unit '{unit}' in duration string '{duration_str}'")

    if total_seconds < 0:
        # This might happen if a negative number was somehow parsed, though regex expects \d+
        # Or if future logic allows negative components. For now, treat as invalid if total is negative.
        raise ValueError(f"Calculated total duration is negative for '{duration_str}'")

    return timedelta(seconds=total_seconds)

def parse_duration_seconds(duration_str):
        """
        Parses a duration string like '1H', '30min', '2D', '1h30m', '2days 5hours' 
        into a timedelta object.
        
        Supports the following units:
            - H, h: hours
            - min, m: minutes
            - D, d, days: days
            - S, s: seconds
            - W, w, weeks: weeks

        Args:
            duration_str (str): The duration string to parse.

        Returns:
            datetime.timedelta: A timedelta object representing the duration.

        Raises:
            ValueError: If the duration string is invalid.
        """


        if duration_str == "0" or duration_str == None or duration_str == "":
            return 0

        pattern = r'(\d+)\s*([HhmindaysSwW]+)'
        matches = re.findall(pattern, duration_str)

        if not matches:
            raise ValueError("Invalid duration format")

        total_seconds = 0
        for value, unit in matches:
            value = int(value)
            if unit in ('H', 'h'):
                total_seconds += value * 3600
            elif unit in ('min', 'm'):
                total_seconds += value * 60
            elif unit in ('D', 'd', 'days'):
                total_seconds += value * 86400
            elif unit in ('S', 's'):
                total_seconds += value
            elif unit in ('W', 'w', 'weeks'):
                total_seconds += value * 604800
            else:
                return 0

        return total_seconds - 30