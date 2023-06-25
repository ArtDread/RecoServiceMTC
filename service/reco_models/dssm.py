from __future__ import annotations

from abc import ABC, abstractmethod

import dill


class DSSM(ABC):
    def __init__(self, name: str):
        with open(f"{name}", "rb") as f:
            self.model = dill.load(f)

    @abstractmethod
    def predict(self, user_id: int) -> None | list[int]:
        pass


class OfflineTDSSM(DSSM):
    def predict(self, user_id: int) -> None | list[int]:
        if user_id in self.model:
            return self.model[user_id]
        return None
