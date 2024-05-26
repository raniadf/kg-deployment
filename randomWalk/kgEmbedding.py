# IMPORT LIBRARIES
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransE, ComplEx, PairRE
import sys
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# IMPORT CUSTOM MODULES
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from databaseHandler import DatabaseHandler

class kgEmbedding:
    def __init__(self, model_name='TransE', embedding_dim=272, num_epochs=25, learning_rate=0.1, batch_size=10):
        self.db = DatabaseHandler()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pipeline_result = None
        self._load_data()
        self._create_model()
        self.db.close()

    def _load_data(self):
        with self.db.driver.session() as session:
            self.records = session.execute_read(self.db.fetch_relations)
        self.create_triplets()

    def create_triplets(self):
        triplets = [(record['source_name'], record['relation'], record['target_name']) for record in self.records]
        self.triples_factory = TriplesFactory.from_labeled_triples(np.array(triplets, dtype=object))

    def _create_model(self):
        if self.model_name == 'TransE':
            model_cls = TransE
        elif self.model_name == 'ComplEx':
            model_cls = ComplEx
        elif self.model_name == 'PairRE':
            model_cls = PairRE
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        self.pipeline_result = pipeline(
            model=model_cls,
            training=self.triples_factory,
            testing=self.triples_factory, 
            model_kwargs={'embedding_dim': self.embedding_dim},
            optimizer_kwargs={'lr': self.learning_rate},
            training_kwargs={'num_epochs': self.num_epochs, 'batch_size': self.batch_size},
            random_seed=42
        )

    def evaluate_model(self):
        if self.pipeline_result:
            return self.pipeline_result.metric_results.to_dict()
        else:
            return {}

# FOR TESTING PURPOSES
if __name__ == "__main__":
    kg = kgEmbedding(model_name='TransE', embedding_dim=100, num_epochs=150, batch_size=32)
    print(kg.pipeline_result)

    kg.db.close()