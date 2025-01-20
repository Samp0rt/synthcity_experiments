# stdlib
from typing import Any, List

# third party
import pandas as pd
import numpy as np

# synthcity absolute
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.plugins.core.distribution import Distribution
from synthcity.plugins.core.plugin import Plugin
from synthcity.plugins.core.schema import Schema
from ganblr.models import GANBLR, GANBLRPP


class GANBLR_plugin(Plugin):

    def __init__(self, cat_limit: int = 15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cat_limit = cat_limit
        self.model = GANBLR()

    @staticmethod
    def name() -> str:
        return "ganblr"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, batch_size=32, epochs=50, warmup_epochs=1, verbose=1, *args: Any, **kwargs: Any) -> "GANBLR_plugin":
        data = X.dataframe().drop(columns=[X.target_column])
        labels = X.dataframe()[X.target_column]

        self.model.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            verbose=verbose
        )
        return self

    def _generate(self, syn_schema: Schema, count: int, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)
    

class GANBLRPP_plugin(Plugin):

    def __init__(self, cat_limit: int = 15, numerical_columns: List[int] = [], random_state: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.numerical_columns = numerical_columns
        self.cat_limit = cat_limit
        self.random_state = random_state
        self.model = GANBLRPP(
            numerical_columns=self.numerical_columns,
            random_state=self.random_state
        )

    @staticmethod
    def name() -> str:
        return "ganblr++"

    @staticmethod
    def type() -> str:
        return "debug"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Distribution]:
        return []

    def _fit(self, X: DataLoader, batch_size=32, epochs=50, warmup_epochs=1, verbose=1, *args: Any, **kwargs: Any) -> "GANBLRPP_plugin":
        data = np.array(X.dataframe().drop(columns=[X.target_column]))
        labels = np.array(X.dataframe()[X.target_column])
        
        self.model.fit(
            x=data,
            y=labels,
            batch_size=batch_size,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            verbose=verbose
        )
        return self

    def _generate(self, syn_schema: Schema, count: int, **kwargs: Any) -> pd.DataFrame:
        return self._safe_generate(self.model.sample, count, syn_schema)