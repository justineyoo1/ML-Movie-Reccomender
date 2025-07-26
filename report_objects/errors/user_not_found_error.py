class UserNotFoundError(ValueError):
    """
    UserNotFoundError class.

    An Error class that handles attempting to get recommendations for a user
    that doesn't exist in the dataset.

    Args:
    ----
        ValueError (Error): The ValueError class that UserNotFoundError extends.

    """

    def __init__(self, message: str):
        """
        UserNotFoundError constructor.

        Initialize a UserNotFoundError object by setting the fields
        to the arguments passed to the constructor.

        Args:
        ----
            message (str): The error message displayed to the user, based
                           on the user lookup failure.

        """
        self.message = message
        super().__init__(self.message) 