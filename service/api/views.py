from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.security import HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from pydantic import BaseModel

from service.api.exceptions import (
    BearerAccessTokenError,
    ModelNotFoundError,
    MultiplicityUserId,
    UserNotFoundError,
)
from service.api.explanation_response import explanationGenerators
from service.api.reco_response import recoGenerators
from service.api.responses import (
    AuthorizationResponse,
    ForbiddenResponse,
    NotFoundError,
)
from service.configuration import (
    AE_PATHS,
    ANN_PATHS,
    LIGHTFM_PATHS,
    OFFLINE_KNN_MODEL,
    ONLINE_KNN_MODEL,
    POPULAR_IN_CATEGORY,
    POPULAR_MODEL_RECS,
    POPULAR_MODEL_USERS,
    TDSSM_PATH,
)
from service.log import app_logger
from service.reco_models import (
    ANNLightFM,
    OfflineKnnModel,
    OnlineAE,
    OnlineKnnModel,
    OnlineLightFM,
    PopularInCategory,
    SimplePopularModel,
    OfflineTDSSM,
)

baseline_model = PopularInCategory(POPULAR_IN_CATEGORY)
popular_model = SimplePopularModel(
    POPULAR_MODEL_USERS,
    POPULAR_MODEL_RECS,
)
offline_knn_model = OfflineKnnModel(OFFLINE_KNN_MODEL)
online_knn_model = OnlineKnnModel(ONLINE_KNN_MODEL)
online_fm_part_popular = OnlineLightFM(LIGHTFM_PATHS)
online_fm_all_popular = OnlineLightFM(LIGHTFM_PATHS, False)
ann_lightfm = ANNLightFM(ANN_PATHS, popular_model)
ae_model = OnlineAE(AE_PATHS)
tdssm = OfflineTDSSM(TDSSM_PATH)


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class ExplainResponse(BaseModel):
    p: int
    explanation: str


bearer_scheme = HTTPBearer()

router = APIRouter()

responses: dict[str, object] = {
    "401": AuthorizationResponse().get_response(),
    "403": ForbiddenResponse().get_response(),
    "404": NotFoundError().get_response(),
}


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=responses,  # type: ignore
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> RecoResponse:
    """
    Get top-k recommendations for the specific user_id using selected model.

    Args:
        request: The request.
        model_name: The name of model to make recommendations.
        user_id: The specific user id for which recommendations are demanded.
        token: The token to verify a service user.

    Returns:
        A RecoResponse containing (user_id, items), where user_id is the same as input
            parameter, items are top-k recommendations.

    """
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if token.credentials != "DanielMoor":
        raise BearerAccessTokenError()
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    emulate_random_error: bool = request.app.state.emulate_random_error
    if emulate_random_error and (user_id and not user_id % 666):
        raise MultiplicityUserId(error_message=f"User {user_id} is a multiple of 666")

    k_recs: int = request.app.state.k_recs
    reco: None | list[int] = None
    try:
        reco = recoGenerators[model_name](k_recs, user_id)
    except KeyError:
        raise ModelNotFoundError(error_message=f"Model {model_name} not found")

    if not reco:
        reco = popular_model.predict(user_id, k_recs)
    return RecoResponse(user_id=user_id, items=reco)


@router.get(
    path="/explain/{model_name}/{user_id}/{item_id}",
    tags=["Explanations"],
    response_model=ExplainResponse,
    responses=responses,  # type: ignore
)
async def explain(
    model_name: str,
    user_id: int,
    item_id: int,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> ExplainResponse:
    """
    Get the explanation for the relevance of (user_id, item_if) for specific model.

    Return the relevance score of specific item for specific user and text explanation
    as well. The method of generating that information will be depend on model type.

    Args:
        model_name: The name of model for which an explanation is demanded.
        user_id: The specific user id for which an explanation is demanded.
        item_id: The specific item id for which an explanation is demanded.
        token: The token to verify a service user.

    Returns:
        An ExplainResponse containing (p, explanation), where p is the item relevance
            score for the provided user, in %; explanation is the explanation message
                of result.

    """
    if token.credentials != "DanielMoor":
        raise BearerAccessTokenError()
    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    try:
        p, explanation = explanationGenerators[model_name](user_id, item_id)
    except KeyError:
        raise ModelNotFoundError(
            error_message=f"Explanation for model {model_name} not implemented yet"
        )
    return ExplainResponse(p=p, explanation=explanation)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
