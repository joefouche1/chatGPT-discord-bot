from typing import Optional, Tuple
from utils.log import logger

# Action code definitions
ACTION_CODES = {
    "!SPORTS": "sports",
    "!SPORTSNOW": "sportsnow",
    "!WEATHER": "weather",
    "!NEWS": "news",
    "!DRAW": "draw"
}

async def process_action_code(message: str) -> Optional[Tuple[str, str]]:
    """Check if message contains an action code and extract parameters."""
    for code in ACTION_CODES:
        # Use find instead of startswith to catch action codes anywhere in the message
        index = message.find(code)
        if index != -1:
            # Extract everything after the action code
            params = message[index + len(code):].strip()
            logger.info(f"Action code detected: {code} with params: {params}")
            return (code, params)
    return None 