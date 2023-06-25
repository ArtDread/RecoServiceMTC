from __future__ import annotations

import service.api.views as views


def simple_recos(k_recs: int, user_id: int) -> list[int]:
    """Generate dummy recos."""
    return list(range(k_recs))


def baseline_recos(k_recs: int, user_id: int) -> list[int]:
    """Use popular-in-category model to generate recos."""
    return views.baseline_model.predict(user_id, k_recs)


def offline_knn_recos(k_recs: int, user_id: int) -> None | list[int]:
    """Use knn model results (dict) to generate recos."""
    return views.offline_knn_model.predict(user_id)


def online_knn_recos(k_recs: int, user_id: int) -> None | list[int]:
    """Use knn model to generate recos on the fly."""
    return views.online_knn_model.predict(user_id)


def light_fm_hot_recos(k_recs: int, user_id: int) -> None | list[int]:
    """Use LightFM model to generate recos for hot users, popular for cold."""
    return views.online_fm_all_popular.predict(user_id, k_recs)


def light_fm_all_recos(k_recs: int, user_id: int) -> None | list[int]:
    """LightFM to generate recos for hot and cold users if possible."""
    return views.online_fm_part_popular.predict(user_id, k_recs)


def ann_lightfm_recos(k_recs: int, user_id: int) -> None | list[int]:
    """Use Approximate Nearest Neighbor algo with LightFM embeddings."""
    return views.ann_lightfm.predict(user_id)


def ae_recos(k_recs: int, user_id: int) -> None | list[int]:
    """Use PyTorch autoencoder model to generate recos."""
    return views.ae_model.predict(user_id, k_recs)


def tdssm_recs(k_recs: int, user_id: int) -> None | list[int]:
    return views.tdssm.predict(user_id)


recoGenerators = {
    "test_model": simple_recos,
    "baseline": baseline_recos,
    "knn": offline_knn_recos,
    "online_knn": online_knn_recos,
    "light_fm_hot": light_fm_hot_recos,
    "light_fm_all": light_fm_all_recos,
    "ann_lightfm": ann_lightfm_recos,
    "ae_model": ae_recos,
    "tdssm": tdssm_recs,
}
