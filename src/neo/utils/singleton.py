class Singleton(type):
    """
    A metaclass that ensures only one instance of a class is created.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        # If an instance doesn't exist, create one and save it in _instances
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        # Return the (only) instance
        return cls._instances[cls]
