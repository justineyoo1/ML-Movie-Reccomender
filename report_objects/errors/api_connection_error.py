class APIConnectionError(ConnectionError):
    """
    APIConnectionError class.

    An Error class that handles failures when connecting to external APIs
    such as TMDB for movie metadata and poster images.

    Args:
    ----
        ConnectionError (Error): The ConnectionError class that APIConnectionError extends.

    """

    def __init__(self, message: str):
        """
        APIConnectionError constructor.

        Initialize an APIConnectionError object by setting the fields
        to the arguments passed to the constructor.

        Args:
        ----
            message (str): The error message displayed to the user, based
                           on the API connection failure.

        """
        self.message = message
        super().__init__(self.message) 