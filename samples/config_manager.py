import os
import json

class ConfigManager:
    """
    Manages loading configuration files in JSON format and handling environment
    variable overrides for sensitive keys.

    Implemented method:
        load() — reads the JSON config from disk and applies env-var overrides.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.settings = {}

    def load(self):
        """Loads configuration from the filesystem."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found at {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.settings = json.load(f)
        
        # Override with environment variables if present, preserving original types.
        for key in self.settings.keys():
            env_key = f"APP_{key.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                orig = self.settings[key]
                try:
                    if isinstance(orig, bool):
                        normalized = raw.strip().lower()
                        if normalized == 'true':
                            coerced = True
                        elif normalized == 'false':
                            coerced = False
                        else:
                            raise ValueError(
                                f"Invalid boolean value for key '{env_key}': "
                                f"expected 'true' or 'false', got {raw!r}"
                            )
                    elif isinstance(orig, int):
                        coerced = int(raw)
                    elif isinstance(orig, float):
                        coerced = float(raw)
                    elif isinstance(orig, (list, dict)):
                        parsed = json.loads(raw)
                        if not isinstance(parsed, type(orig)):
                            raise TypeError("type mismatch")
                        coerced = parsed
                    else:
                        coerced = raw
                except Exception:
                    coerced = raw
                self.settings[key] = coerced

    def get(self, key, default=None):
        """Retrieves a setting by key."""
        return self.settings.get(key, default)

def validate_email(email):
    """Simple email validation logic using string splits."""
    email = email.strip()
    if "@" not in email:
        return False
    parts = email.split("@")
    if len(parts) != 2:
        return False
    local, domain = parts
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    labels = domain.split(".")
    if any(label == "" for label in labels):
        return False
    return True
