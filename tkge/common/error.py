class ConfigurationError(Exception):
    """Exception for error cases in the `tkge.common.config.Config` class."""
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg
