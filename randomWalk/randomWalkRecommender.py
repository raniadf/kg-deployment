import numpy as np
from .randomWalk import randomWalk
from .kgEmbedding import kgEmbedding
import json

class RandomWalkRecommender(randomWalk):
    def __init__(self, kg_embedding, pretest_results_path, pass_threshold=7.5):
        super().__init__(kg_embedding)
        # self.pretest_results = self.load_pretest_results(pretest_results_path) # For manual testing
        self.pretest_results = pretest_results_path # For integrated testing
        self.pass_threshold = pass_threshold
        self.transition_matrix = None

    def load_pretest_results(self, file_path):
        with open(file_path, 'r') as file:
            pretest_results = json.load(file)
        return pretest_results
    
    def identify_passed_courses(self):
        print(self.pretest_results)
        return [course['course_id'] for course in self.pretest_results if course['score'] >= self.pass_threshold]

    def recommend_courses(self):
        self.build_transition_matrix()
        transition_matrix = self.transition_matrix.toarray()
        for row in transition_matrix:
            print(row)
        
        # Identifikasi modul yang sudah lulus
        passed_courses = self.identify_passed_courses()
        num_courses = len(self.pretest_results)
        
        # Membentuk matriks probabilitas awal dengan nilai 1 untuk semua modul
        probabilities = np.ones(num_courses)

        # Set probabilitas 0 untuk modul yang sudah lulus
        for course_id in passed_courses:
            probabilities[course_id - 1] = 0 
        
        # Melakukan random walk sebanyak 100 kali
        for _ in range(100):  
            probabilities = probabilities @ transition_matrix
        
        print("Final Probabilities")
        print(probabilities)
        
        recommendations = np.argsort(-probabilities)
        recommended_courses = [int(course_id + 1) for course_id in recommendations if probabilities[course_id] > 0 and self.pretest_results[course_id]['score'] < self.pass_threshold]
        recommended_courses.sort()

        return recommended_courses

if __name__ == "__main__":
    pretest_results_path = './data/pretestResults.json'
    pretest_results_path_2 = './data/pretestResults2.json'
    kg_embedding = kgEmbedding()
    recommender = RandomWalkRecommender(kg_embedding, pretest_results_path)
    learning_path_recommendations = recommender.recommend_courses()
    print(learning_path_recommendations)