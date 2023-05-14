from __future__ import annotations

import service.api.views as views


def popular_explanation(user_id: int, item_id: int) -> tuple[int, str]:
    """Generate relevance score and explanation using popular model."""
    return views.popular_model.explain(user_id, item_id)


def online_knn_explanation(user_id: int, item_id: int) -> tuple[int, str]:
    """Generate relevance score and explanation using online knn model.

    If the user is hot, try to find the item amongst the similar users and calculate
    relevance score based on his popularity.
    If the user is cold, return the explanation result from model that handles the cold.
    """
    # information: tuple[int, str] | None = views.online_knn_model.explain(
    #     user_id, item_id
    # )
    information = None
    # If user is cold get the explanation from popular model
    return information if information else popular_explanation(user_id, item_id)


explanationGenerators = {
    "popular_model": popular_explanation,
    "online_knn": online_knn_explanation,
}
