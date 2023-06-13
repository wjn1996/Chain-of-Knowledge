import os
import json

"""
F^2 Vertification
- factuality verifying
- faithfulness verifying
"""
class Factuality(object):

    def __init__(self, args) -> None:
        self.args = args
        # load pre-trained kg embeddings
        self.entity_embeddings, self.relation_embeddings = self.load_embeddings()
    
    def load_embeddings(self):
        return None, None

    def calculate_factuality_score(self, evidence_triples: list):
        for triple in evidence_triples:
            subject, relation, object = triple
            