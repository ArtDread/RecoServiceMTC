from __future__ import annotations

import json
from http import HTTPStatus

import orjson
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from service.models import Error


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object) -> object:
        if isinstance(o, BaseModel):
            return o.dict()
        try:
            orjson.dumps(o)
        except TypeError:
            return str(o)
        return super().default(o)


class DataclassJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: object) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
            cls=EnhancedJSONEncoder,
        ).encode("utf-8")


def create_response(
    status_code: int,
    message: None | str = None,
    data: None | object = None,
    errors: None | list[Error] = None,
) -> JSONResponse:
    content: dict[str, object] = {}

    if message:
        content["message"] = message

    if data:
        content["data"] = data

    if errors:
        content["errors"] = errors

    return DataclassJSONResponse(content, status_code=status_code)


def server_error(errors: list[Error]) -> JSONResponse:
    return create_response(HTTPStatus.INTERNAL_SERVER_ERROR, errors=errors)
