from __future__ import annotations

import dill
import nmslib
import numpy as np
from lightfm import LightFM
from numpy.typing import NDArray
from scipy import sparse

from .popular import SimplePopularModel


class OnlineLightFM:
    """This class is implementation of recommendations generation with LightFM.

    LightFM library realization is utilized. We use built-in method predict()
    to generate recos both for hot users — i.e. who has interactions — and
    cold users — i.e. who could possibly have only features. If cold user
    has no features at all then popular model is the best option to make
    recommendation.

    Attributes:
        model (LightFM): The instance of the LightFM class.
        user_mapping (dict[int, int]): The dictionary to make the transition from
            internal (generated during the model fitting) to external user ids.
        item_mapping (dict[int, int]): The dictionary to make the transition from
            external to internal item ids.
        features_for_cold (dict[int, dict[str, str]]): The features values for every
            known cold user.
        unique_features (ndarray[np.unicode_]): The all possible user features values.
        items_internal_ids (ndarray[np.int64]): The numpy array containing internal item
            ids.
        cold_with_fm (bool): The flag to use LightFM model to to generate recos
            for cold users having features or not (use popular instead).

    """

    def __init__(
        self,
        lightfm_paths: tuple[str, str, str, str, str],
        cold_with_fm: bool = True,
    ):
        (
            model_path,
            user_mapping_path,
            item_mapping_path,
            features_for_cold_path,
            unique_features_path,
        ) = lightfm_paths
        try:
            with open(f"{model_path}", "rb") as f:
                self.model: LightFM = dill.load(f)
        except FileNotFoundError:
            print("Run `make load_models` to load the pickled object")

        with open(user_mapping_path, "rb") as f:
            self.user_mapping: dict[int, int] = dill.load(f)
        with open(item_mapping_path, "rb") as f:
            self.item_mapping: dict[int, int] = dill.load(f)
        with open(features_for_cold_path, "rb") as f:
            self.features_for_cold: dict[int, dict[str, str]] = dill.load(f)
        with open(unique_features_path, "rb") as f:
            self.features: NDArray[np.unicode_] = dill.load(f)

        self.items_internal_ids = np.arange(
            len(self.item_mapping.keys()), dtype=np.int64
        )
        self.cold_with_fm = cold_with_fm

    def _get_hot_reco(self, iternal_user_id: int, k_recs: int) -> list[int]:
        hot_scores: NDArray[np.float32] = self.model.predict(
            iternal_user_id, item_ids=self.items_internal_ids
        )
        idxs = np.argsort(hot_scores)[::-1]
        recs = self.items_internal_ids[idxs][:k_recs]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs

    def _get_cold_reco(self, user_feature: dict[str, str], k_recs: int) -> list[int]:
        user_feature_list = list(user_feature.values())
        feature_mask = np.isin(self.features, user_feature_list)
        feature_row = sparse.csr_matrix(feature_mask)

        cold_scores: NDArray[np.float32] = self.model.predict(
            0, self.items_internal_ids, user_features=feature_row
        )
        idxs = np.argsort(cold_scores)[::-1]
        recs = self.items_internal_ids[idxs][:k_recs]
        recs = [self.item_mapping[reco] for reco in recs]
        return recs

    def predict(self, user_id: int, k_recs: int) -> None | list[int]:
        # Check if user is hot or not
        iternal_user_id = self.user_mapping.get(user_id, None)
        if iternal_user_id:
            return self._get_hot_reco(iternal_user_id=iternal_user_id, k_recs=k_recs)

        if self.cold_with_fm:
            # Check if cold user have any features
            user_feature = self.features_for_cold.get(user_id, None)
            if user_feature:
                return self._get_cold_reco(user_feature=user_feature, k_recs=k_recs)
        # If not the case, let the popular model make recos
        return None


class ANNLightFM:
    # pylint: disable=too-many-instance-attributes
    # Eight is reasonable in this case.
    def __init__(
        self,
        ann_paths: tuple[str, str, str, str, str, str],
        popular_model: SimplePopularModel,
        k: int = 10,
    ):
        (
            user_m,
            item_inv_m,
            index_path,
            user_emb,
            watched_u2i,
            cold_reco_dict,
        ) = ann_paths
        self.K = k
        with open(user_m, "rb") as f:
            self.user_m: dict[int, int] = dill.load(f)
        with open(item_inv_m, "rb") as f:
            self.item_inv_m: dict[int, int] = dill.load(f)
        self.index = nmslib.init(method="hnsw", space="negdotprod")
        self.index.loadIndex(index_path, load_data=True)
        try:
            with open(user_emb, "rb") as f:
                self.user_emb: NDArray[np.float32] = dill.load(f)
        except FileNotFoundError:
            print("Run `make load_models` to load a pickled object")
        with open(watched_u2i, "rb") as f:
            self.watched_u2i: dict[int, list[int]] = dill.load(f)
        with open(cold_reco_dict, "rb") as f:
            self.cold_reco_dict: dict[int, list[int]] = dill.load(f)
        self.popular_model: SimplePopularModel = popular_model

    def predict(self, user_id: int) -> None | list[int]:
        if user_id in self.user_m:
            user_vector = self.user_emb[self.user_m[user_id]]
            pr_internal_items = self.index.knnQuery(vector=user_vector, k=self.K)[0]
            pr_items = [self.item_inv_m[item] for item in pr_internal_items]

            # Delete already seen items
            pr_items_numpy = np.array(pr_items, dtype="uint16")
            already_seen_items = np.array(self.watched_u2i[user_id], dtype="uint16")

            unseen_items = pr_items_numpy[~np.isin(pr_items_numpy, already_seen_items)]
            num_lost_items = self.K - unseen_items.shape[0]
            if num_lost_items > 0:
                popular_items = np.array(
                    self.popular_model.predict(user_id, 5 * self.K)
                )

                popular_items = popular_items[
                    ~np.isin(popular_items, already_seen_items)
                ]
                popular_items = popular_items[~np.isin(popular_items, unseen_items)]

                unseen_items = np.append(unseen_items, popular_items[:num_lost_items])
                if len(unseen_items) != 10:
                    return self.popular_model.predict(user_id, k_recs=self.K)
            return unseen_items[: self.K].tolist()
        return self.popular_model.predict(user_id, k_recs=self.K)
