"""This module contains paths to the saved python objects."""

# Popular models and its data
POPULAR_MODEL_RECS = "models/popular/popular_dictionary.pickle"
POPULAR_MODEL_USERS = "models/popular/users_dictionary.pickle"
POPULAR_IN_CATEGORY = "models/popular/popular_in_category.dill"

# KNN models and its data
OFFLINE_KNN_MODEL = "models/knn/dictionary_with_hot_recs.dill"
ONLINE_KNN_MODEL = "models/knn/user_knn.dill"

# Factorization Machines models and its data
LIGHT_FM = "models/lightfm/light_fm.dill"
USER_MAPPING = "models/lightfm/user_mapping.dill"
ITEM_MAPPING = "models/lightfm/item_mapping.dill"
FEATURES_FOR_COLD = "models/lightfm/features_for_cold.dill"
UNIQUE_FEATURES = "models/lightfm/unique_features.dill"

LIGHTFM_PATHS = (
    LIGHT_FM,
    USER_MAPPING,
    ITEM_MAPPING,
    FEATURES_FOR_COLD,
    UNIQUE_FEATURES,
)

ANN_USER_MAPPING = "models/lightfm/ann/user_mapping.dill"
ANN_ITEM_INV_MAPPING = "models/lightfm/ann/item_inv_mapping.dill"
ANN_ITEMS_INDEX = "models/lightfm/ann/items_index.hnsw"
ANN_USER_EMBEDDING = "models/lightfm/ann/user_embeddings.dill"
ANN_WATCHED_U2I_DICT = "models/lightfm/ann/watched_user2items_dictionary.dill"
ANN_COLD_RECO_DICT = "models/lightfm/ann/cold_users_dictionary_popular.dill"

ANN_PATHS = (
    ANN_USER_MAPPING,
    ANN_ITEM_INV_MAPPING,
    ANN_ITEMS_INDEX,
    ANN_USER_EMBEDDING,
    ANN_WATCHED_U2I_DICT,
    ANN_COLD_RECO_DICT,
)

# AE models and its data
AE_MODEL = "models/ae/VariationalAE"
AE_HOT_USERS = "models/ae/hot_users.dill"
AE_HOT_USERS_WEIGHTS = "models/ae/hot_users_weights.dill"
AE_ITEM_MAPPING = "models/ae/item_to_ind"
AE_ENC_DIMS = "models/ae/VariationalAE_dims.pickle"

AE_PATHS = (
    AE_MODEL,
    AE_HOT_USERS,
    AE_HOT_USERS_WEIGHTS,
    AE_ITEM_MAPPING,
    AE_ENC_DIMS,
)
