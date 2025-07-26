class ModelNotTrainedError(RuntimeError):
    """
    ModelNotTrainedError class.

    An Error class that handles attempting to use a model for predictions
    when it hasn't been trained yet.

    Args:
    ----
        RuntimeError (Error): The RuntimeError class that ModelNotTrainedError extends.

    """

    def __init__(self, message: str):
        """
        ModelNotTrainedError constructor.

        Initialize a ModelNotTrainedError object by setting the fields
        to the arguments passed to the constructor.

        Args:
        ----
            message (str): The error message displayed to the user, based
                           on the model state.

        """
        self.message = message
        super().__init__(self.message) 