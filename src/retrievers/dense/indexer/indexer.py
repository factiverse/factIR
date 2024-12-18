"""
 FAISS-based index components for dense retriver
"""

import logging

from src.retrievers.dense.indexer.annoyIndexer import AnnoySearch
from src.retrievers.dense.indexer.faissIndexer import FaissSearch


logger = logging.getLogger()

from src.config import AnnConstants


class AnnSearch:
    @staticmethod
    def get_ann_instance(ann_class, data, emb_dim, num_trees=None, metric=None):
        if ann_class.lower() == AnnConstants.FAISS_SEARCH.lower():
            return FaissSearch(data, emb_dim)

        elif ann_class.lower() == AnnConstants.ANNOY_SEARCH.lower():
            return AnnoySearch(data, num_trees, emb_dim, metric)

        else:
            classname = ann_class
            if classname not in globals():
                raise ValueError(
                    "No implementation found for the custom data generator class specified: {}".format(
                        classname
                    )
                )
            return globals()[classname]()
