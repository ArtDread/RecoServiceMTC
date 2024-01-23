from http import HTTPStatus

from starlette.testclient import TestClient

from service.settings import ServiceConfig

GET_RECO_PATH = "/reco/{model_name}/{user_id}"
GET_EXPLAIN_PATH = "/explain/{model_name}/{user_id}/{item_id}"


def test_health(
    client: TestClient,
) -> None:
    with client:
        response = client.get("/health")
    assert response.status_code == HTTPStatus.OK


def test_get_reco_success(
    client: TestClient,
    service_config: ServiceConfig,
) -> None:
    user_id = 123
    path = GET_RECO_PATH.format(model_name="test_model", user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": "Bearer DanielMoor"})
    assert response.status_code == HTTPStatus.OK
    response_json = response.json()
    assert response_json["user_id"] == user_id
    assert len(response_json["items"]) == service_config.k_recs
    assert all(isinstance(item_id, int) for item_id in response_json["items"])


def test_get_reco_for_unknown_user(
    client: TestClient,
) -> None:
    user_id = 10**10
    path = GET_RECO_PATH.format(model_name="test_model", user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": "Bearer DanielMoor"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_reco_for_unknown_model(
    client: TestClient,
) -> None:
    user_id = 123
    incorrect_model = "_"
    path = GET_RECO_PATH.format(model_name=incorrect_model, user_id=user_id)
    with client:
        response = client.get(path, headers={"Authorization": "Bearer DanielMoor"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"


def test_bearer_reco_failed(
    client: TestClient,
) -> None:
    user_id = 123
    incorrect_bearer = "lasdkladsk"
    path = GET_RECO_PATH.format(model_name="test_model", user_id=user_id)
    with client:
        response = client.get(
            path, headers={"Authorization": f"Bearer {incorrect_bearer}"}
        )
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["errors"][0]["error_key"] == "incorrect_bearer_key"


def test_get_explain_popular_relevance_success(
    client: TestClient,
) -> None:
    user_id = 272
    item_ids = [14488, 4151, 3936, 496]  # The corresponding ranks [1, 5, 10, 18]
    ps = [97, 87, 76, 57]  # The relevance scores
    for i, item_id in enumerate(item_ids):
        path = GET_EXPLAIN_PATH.format(
            model_name="popular_model", user_id=user_id, item_id=item_id
        )
        with client:
            response = client.get(path, headers={"Authorization": "Bearer DanielMoor"})
        assert response.status_code == HTTPStatus.OK
        response_json = response.json()
        assert response_json["p"] == ps[i]


def test_get_explain_popular_explanation_success(
    client: TestClient,
) -> None:
    user_ids = [272, 2, 2]
    item_ids = [14488, 5543, 565]
    age_low, age_high, income_low, income_high, sex, kids = (
        45,
        54,
        20,
        40,
        "male",
        "having kids",
    )
    explain_messages = [
        (
            "This item is on top of the popular items amongst users "
            f"from similar contingent: age in {age_low}-{age_high}, income in "
            f"{income_low}-{income_high}, {sex}, {kids}. You probably would "
            "love it"
        ),
        (
            "This item is on top of the popular items amongst all users. "
            "We are higly recommend to check it out!"
        ),
        (
            "This item is not on top of the popular items so we are not sure "
            "about this recommendation. It's up to you to decide"
        ),
    ]
    for i, _ in enumerate(user_ids):
        path = GET_EXPLAIN_PATH.format(
            model_name="popular_model", user_id=user_ids[i], item_id=item_ids[i]
        )
        with client:
            response = client.get(path, headers={"Authorization": "Bearer DanielMoor"})
        assert response.status_code == HTTPStatus.OK
        response_json = response.json()
        assert response_json["explanation"] == explain_messages[i]


def test_get_explain_for_unknown_user(
    client: TestClient,
) -> None:
    user_id, item_id = 10**10, 14488
    path = GET_EXPLAIN_PATH.format(
        model_name="popular_model", user_id=user_id, item_id=item_id
    )
    with client:
        response = client.get(path, headers={"Authorization": "Bearer DanielMoor"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "user_not_found"


def test_get_explain_for_unknown_model(
    client: TestClient,
) -> None:
    user_id, item_id = 123, 14488
    incorrect_model = "_"
    path = GET_EXPLAIN_PATH.format(
        model_name=incorrect_model, user_id=user_id, item_id=item_id
    )
    with client:
        response = client.get(path, headers={"Authorization": "Bearer DanielMoor"})
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json()["errors"][0]["error_key"] == "model_not_found"


def test_bearer_explain_failed(
    client: TestClient,
) -> None:
    user_id, item_id = 123, 14488
    incorrect_bearer = "lasdkladsk"
    path = GET_EXPLAIN_PATH.format(
        model_name="popular_model", user_id=user_id, item_id=item_id
    )
    with client:
        response = client.get(
            path, headers={"Authorization": f"Bearer {incorrect_bearer}"}
        )
    assert response.status_code == HTTPStatus.UNAUTHORIZED
    assert response.json()["errors"][0]["error_key"] == "incorrect_bearer_key"
