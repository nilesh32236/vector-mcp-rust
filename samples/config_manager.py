import os
import json

class ConfigManager:
    """
    Manages loading and saving configuration files in JSON format.
    Handles environment variable overrides for sensitive keys.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.settings = {}

    def load(self):
        """Loads configuration from the filesystem."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.settings = json.load(f)
        
        # Override with environment variables if present
        for key in self.settings.keys():
            env_key = f"APP_{key.upper()}"
            if env_key in os.environ:
                self.settings[key] = os.environ[env_key]

    def get(self, key, default=None):
        """Retrieves a setting by key."""
        return self.settings.get(key, default)

def validate_email(email):
    """Simple email validation logic using string splits."""
    if "@" not in email:
        return False
    parts = email.split("@")
    return len(parts) == 2 and "." in parts[1]
