# src/app.py
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from randomWalk.randomWalkRecommender import RandomWalkRecommender
from randomWalk.randomWalk import visualize_graph_with_weights
from randomWalk.kgEmbedding import kgEmbedding
from semanticSimilarity.semanticSimilarityRecommender import SemanticSimilarityRecommender

import json
import nltk



app = Flask(__name__)
CORS(app, support_credentials=True, resources={r"*": {"origins": "*"}})

@app.route('/recommend', methods=['POST', 'OPTIONS'])
def recommend_courses():
    if request.method == 'OPTIONS':
        # Specific CORS headers for a successful preflight request
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Methods', 'GET, HEAD, POST, OPTIONS, PUT')
        response.headers.add('Access-Control-Allow-Headers', 'Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers')
        return response
    elif request.method == 'POST':
        data = request.json
        # print(data)
        pretest_results = data['pretestResults']
        print("Pretest results: ")
        print(pretest_results)

        # Semantic Similarity
        print("Calculating semantic similarity...")
        recommender1 = SemanticSimilarityRecommender(pretest_results)
        learning_path_recommendations1 = recommender1.get_recommended_courses()
        print(learning_path_recommendations1)

        # Random Walk with KG Embedding
        print("Loading KG embedding...")
        kg_embedding = kgEmbedding(model_name='TransE', embedding_dim=100, num_epochs=300, batch_size=32)
        print("KG embedding loaded.")
        recommender2 = RandomWalkRecommender(kg_embedding, pretest_results)
        learning_path_recommendations2 = recommender2.recommend_courses()
        print(learning_path_recommendations2)
                # Visualize the graph with weights
        # print(recommender2.kg_embedding)
        # print(recommender2.transition_matrix)
        # visualize_graph_with_weights(recommender2.kg_embedding, recommender2.transition_matrix)
        recommendations = {
            "recommendationPaths1": learning_path_recommendations1,
            "recommendationPaths2": learning_path_recommendations2
        }
        response = json.dumps(recommendations)
        print(response)

        # jsonMock = {
        #     "recommendations": [0,1,2]
        # }

        return response

if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    app.run(debug = True, host='0.0.0.0', port=8080)
