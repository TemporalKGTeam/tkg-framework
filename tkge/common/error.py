
class ConfigurationError(Exception):
    def __init__(self, msg:str):
        super().__init__()
        self.msg = msg

    def __str__(self):
        return self.msg