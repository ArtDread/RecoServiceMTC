from __future__ import annotations

from abc import ABC, abstractmethod

import dill


class KnnModel(ABC):
    def __init__(self, name: str):
        with open(f"{name}", "rb") as f:
            self.model = dill.load(f)

    @abstractmethod
    def predict(self, user_id: int) -> None | list[int]:
        pass


class OfflineKnnModel(KnnModel):
    def predict(self, user_id: int) -> None | list[int]:
        if user_id in self.model.keys():
            return self.model[user_id]
        return None


class OnlineKnnModel(KnnModel):
    def predict(self, user_id: int) -> None | list[int]:
        return self.model.predict(user_id)
