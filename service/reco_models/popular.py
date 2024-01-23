from __future__ import annotations

import pickle
from typing import TypedDict

import dill


class PopularDict(TypedDict):
    user_to_watched_items_map: dict[int, set[int]]
    user_to_category_map: dict[int, str]
    category_to_popular_recs: dict[str, list[int]]


def rank_to_relevance(rank: int, max_rank: int) -> int:
    """Scheme to convert popular item value to interpretive relevance score.

    Convert sorted by popularity items to some relevance value is not that simple
    and even ambiguous. Probably, the relevance score is completely unrepresentative
    in case of popular model so the explanation message is more important then. We
    could try to assign some numbers under some assumptions.

    Here some heuristics:
    - More popular item - more relevant. We could use ranks and frequency of occurrence
        of item. For the latter popular model must be rebuild.
    - Consider all popular items rather relevant (items loved by many people must be
        okay or even great), i.e. min relevance > 50 %.

    Thus we simply could try apply min-max normalization with new max and min. In
    future, by adding frequencies, relevance score could be more complex and making
    more sense in the end.
    """
    new_min = 97
    new_max = 53
    return int((rank - 1) / (max_rank - 1) * (new_max - new_min) + new_min)


class SimplePopularModel:
    def __init__(self, users_path: str, recs_path: str, scheme=rank_to_relevance):
        self.users_dictionary: dict[int, str] = pickle.load(open(users_path, "rb"))
        self.popular_dictionary: dict[str, list[int]] = pickle.load(
            open(recs_path, "rb")
        )
        self.scheme = scheme
        self.max_rank = len(self.popular_dictionary["popular_for_all"])

    def predict(self, user_id: int, k_recs: int) -> list[int]:
        """
        Returns top-k recommendations for the specific user_id.

        Args:
            user_id: The user's id from the KION dataset.
            k_recs: The number of recos (k) which are considered.

        Returns:
            list[int]: The top-k recs.

        """
        try:
            # Check if user is suitable for category reco
            category = self.users_dictionary.get(user_id, None)
            if category:
                return self.popular_dictionary[category][:k_recs]
            # If not the case, give him popular on average
            return self.popular_dictionary["popular_for_all"][:k_recs]
        except TypeError:
            return list(range(k_recs))

    def explain(self, user_id: int, item_id: int) -> tuple[int, str]:
        """
        Get the explanation for the relevance of (user_id, item_if).

        Based on items popularity amongst the different user categories,
        provide the explanation for the user why this specific item could be
        interesting (or not) to him.

        Args:
            user_id: The user's id from the KION dataset.
            item_id: The item's id from the KION dataset.

        Returns:
            A tuple (p, explanation), where p is the item relevance score for the
                provided user, in %; explanation is the explanation message of result.

        """
        # Check if user is suitable for category reco
        category = self.users_dictionary.get(user_id, None)
        if category:
            # Check limited list of popular items for this category
            popular_items = self.popular_dictionary[category]
            if item_id in popular_items:
                (
                    _,
                    age_low,
                    age_high,
                    _,
                    income_low,
                    income_high,
                    sex,
                    kids,
                ) = category.split("_")
                rank = popular_items.index(item_id) + 1
                p = self.scheme(rank, self.max_rank)

                sex = "male" if sex == "М" else "female"
                kids = "having kids" if kids else "no kids"
                explanation = (
                    "This item is on top of the popular items amongst users "
                    f"from similar contingent: age in {age_low}-{age_high}, income in "
                    f"{income_low}-{income_high}, {sex}, {kids}. You probably would "
                    "love it"
                )

                return p, explanation
        # If not the case, check the popular on average
        popular_items = self.popular_dictionary["popular_for_all"]
        if item_id in popular_items:
            rank = popular_items.index(item_id)
            p = self.scheme(rank, self.max_rank)
            explanation = (
                "This item is on top of the popular items amongst all users. "
                "We are higly recommend to check it out!"
            )
        else:
            # Consider for the rest fifty-fifty relevance (ignorance of real relevance)
            p = 50
            explanation = (
                "This item is not on top of the popular items so we are not sure "
                "about this recommendation. It's up to you to decide"
            )
        return p, explanation


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
                self.model: PopularDict = dill.load(file)
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
