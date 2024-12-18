"""Data Loader factory to map dataset alias to corresponding Data loader class.

Yields:
    Dataloader class corresponding to alias mentioned in config
"""

from config.constants import Dataset, Split
from RegularClaimsLoader import RegularClaimsLoader


class DataLoaderFactory:
    """Data Loader factory to map dataset alias to corresponding Data loader class."""

    def create_dataloader(
        self,
        dataloader_name: str,
        tokenizer="bert-base-uncased",
        config_path="test_config.ini",
        split=Split.TRAIN,
        batch_size=None,
        corpus=None,
    ) -> RegularClaimsLoader:
        if Dataset.factiverse in dataloader_name:
            loader = RegularClaimsLoader
        else:
            raise NotImplemented(f"{dataloader_name} not implemented yet.")
        return loader(
            dataset=dataloader_name,
            config_path=config_path,
            split=split,
            batch_size=batch_size,
            tokenizer=tokenizer,
            corpus=corpus,
        )
