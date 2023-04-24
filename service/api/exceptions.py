import typing as tp
from http import HTTPStatus


class AppException(Exception):
    def __init__(
        self,
        status_code: int,
        error_key: str,
        error_message: str = "",
        error_loc: tp.Optional[tp.Sequence[str]] = None,
    ) -> None:
        self.error_key = error_key
        self.error_message = error_message
        self.error_loc = error_loc
        self.status_code = status_code
        super().__init__()


class UserNotFoundError(AppException):
    def __init__(
        self,
        status_code: int = HTTPStatus.NOT_FOUND,
        error_key: str = "user_not_found",
        error_message: str = "User is unknown",
        error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)


class ModelNotFoundError(AppException):
    def __init__(
        self,
        status_code: int = HTTPStatus.NOT_FOUND,
        error_key: str = "model_not_found",
        error_message: str = "Model is unknown",
        error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)


class BearerAccessTokenError(AppException):
    def __init__(
        self,
        status_code: int = HTTPStatus.UNAUTHORIZED,
        error_key: str = "incorrect_bearer_key",
        error_message: str = "Authorization failure due to incorrect token",
        error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)


class MultiplicityUserId(AppException):
    """Silly error to test bug-tracker Sentry."""

    def __init__(
        self,
        status_code: int = HTTPStatus.BAD_REQUEST,
        error_key: str = "user_mutiple_666",
        error_message: str = "This particular user id is a multiple of 666",
        error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)
