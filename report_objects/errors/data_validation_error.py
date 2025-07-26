class DataValidationError(ValueError):
    """
    DataValidationError class.

    An Error class that handles data validation issues such as missing columns,
    invalid data types, or corrupted data in the MovieLens dataset.

    Args:
    ----
        ValueError (Error): The ValueError class that DataValidationError extends.

    """

    def __init__(self, message: str):
        """
        DataValidationError constructor.

        Initialize a DataValidationError object by setting the fields
        to the arguments passed to the constructor.

        Args:
        ----
            message (str): The error message displayed to the user, based
                           on the data validation failure.

        """
        self.message = message
        super().__init__(self.message) 