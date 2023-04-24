from __future__ import annotations

import pickle

import dill


class SimplePopularModel:
    def __init__(self, users_path: str, recs_path: str):
        self.users_dictionary: dict[int, str] = pickle.load(open(users_path, "rb"))
        self.popular_dictionary: dict[str, list[int]] = pickle.load(
            open(recs_path, "rb")
        )

    def predict(self, user_id: int, k_recs: int) -> list[int]:
        try:
            # Check if user is suitable for category reco
            category = self.users_dictionary.get(user_id, None)
            if category:
                return self.popular_dictionary[category][:k_recs]
            # If not the case, give him popular on average
            return self.popular_dictionary["popular_for_all"][:k_recs]
        except TypeError:
            return list(range(k_recs))


class PopularInCategory:
    """This class is implementation of recommendations generation with
    popular model by user category.

    Attributes:
        model_path (str): The path to pickled model.

    """

    __slots__ = {"model"}

    def __init__(self, model_path: str):
        try:
            with open(model_path, "rb") as file:
                self.model: dict[str, object] = dill.load(file)
        except FileNotFoundError as e:
            print(
                f"ERROR while loading model: {e}"
                f"\nRun `make load_models` to load model from GDrive"
            )

    def predict(self, user_id: int, k: int) -> list[int]:
        """
        Returns top k items for specific user_id.

        Args:
            user_id (int): The user's id from KION dataset.
            k (int): The number of item_ids for that user_id.

        Returns:
            list[int]: k item_ids.

        """
        user_to_watched_items_map: dict[int, set[int]] = self.model[
            "user_to_watched_items_map"
        ]
        user_to_category_map: dict[int, str] = self.model["user_to_category_map"]
        category_to_popular_recs: dict[str, list[int]] = self.model[
            "category_to_popular_recs"
        ]

        watched_items = set()
        if user_id in user_to_watched_items_map:
            watched_items = user_to_watched_items_map[user_id]

        user_category = "default"
        if user_id in user_to_category_map:
            user_category = user_to_category_map[user_id]

        recs_for_user_category = category_to_popular_recs[user_category]
        result = []
        current_recs_in_result = 0
        for item_id in recs_for_user_category:
            if item_id not in watched_items:
                result.append(item_id)
                current_recs_in_result += 1
            if current_recs_in_result == k:
                return result

        recs_default = category_to_popular_recs["default"]
        for item_id in recs_default:
            if item_id not in watched_items and item_id not in result:
                result.append(item_id)
                current_recs_in_result += 1
            if current_recs_in_result == k:
                return result
        return result + [item_id + 1 for item_id in range(k - len(result))]
