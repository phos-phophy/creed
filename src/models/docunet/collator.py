from typing import Dict, List

from src.abstract import CollatedFeatures, Collator, PreparedDocument


class DocUNetCollator(Collator):
    @classmethod
    def collate_fn(cls, documents: List[PreparedDocument]) -> Dict[str, CollatedFeatures]:
        document_dict = {}

        for field_name in documents[0]._fields:

            if documents[0].__getattribute__(field_name) is None:
                continue

            feature_names = documents[0].__getattribute__(field_name).keys()

            for feature_name in feature_names:
                features = cls.get_features(documents, field_name, feature_name)

                if feature_names not in ("labels", "hts", "entity_pos"):
                    features = cls.collate(features)

                document_dict[feature_name] = features

        return document_dict
