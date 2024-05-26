from .semanticSimilarity import SemanticSimilarity
import json
import numpy as np

class SemanticSimilarityRecommender(SemanticSimilarity):
    def __init__(self, pretest_results_path, pass_threshold=7):
        super().__init__()
        # self.pretest_results = self.load_pretest_results(pretest_results_path) # For manual testing
        self.pretest_results = pretest_results_path # For integrated testing
        self.pass_threshold = pass_threshold
    
    def load_pretest_results(self, file_path):
        with open(file_path, 'r') as file:
            pretest_results = json.load(file)
        return pretest_results
    
    def get_recommended_courses(self):
        # similarity_matrix = self.calculate_semantic_similarity()
        similarity_matrix = {(1.0, 3.0): 0, (1.0, 4.0): 0, (1.0, 5.0): 0, (1.0, 6.0): 0, (1.0, 7.0): 0, (1.0, 8.0): 0, (1.0, 9.0): 0, (1.0, 10.0): 0, (1.0, 11.0): 0, (1.0, 2.0): 0, (1.0, 12.0): 0, (1.0, 13.0): 0, (3.0, 1.0): 0, (3.0, 4.0): 0, (3.0, 5.0): 0, (3.0, 6.0): 0, (3.0, 7.0): 0, (3.0, 8.0): 0, (3.0, 9.0): 0, (3.0, 10.0): 0, (3.0, 11.0): 0, (3.0, 2.0): 0.5, (3.0, 12.0): 0, (3.0, 13.0): 0, (4.0, 1.0): 0, (4.0, 3.0): 0.5, (4.0, 5.0): 0, (4.0, 6.0): 0, (4.0, 7.0): 0, (4.0, 8.0): 0, (4.0, 9.0): 0, (4.0, 10.0): 0, (4.0, 11.0): 0, (4.0, 2.0): 0.9227832045442305, (4.0, 12.0): 0, (4.0, 13.0): 0, (5.0, 1.0): 0, (5.0, 3.0): 0.5, (5.0, 4.0): 0.5, (5.0, 6.0): 0, (5.0, 7.0): 0, (5.0, 8.0): 0, (5.0, 9.0): 0, (5.0, 10.0): 0, (5.0, 11.0): 0, (5.0, 2.0): 0.9227832045442305, (5.0, 12.0): 0, (5.0, 13.0): 0, (6.0, 1.0): 0, (6.0, 3.0): 0.5, (6.0, 4.0): 0.3333333333333333, (6.0, 5.0): 0.5, (6.0, 7.0): 0, (6.0, 8.0): 0, (6.0, 9.0): 0, (6.0, 10.0): 0, (6.0, 11.0): 0, (6.0, 2.0): 0.9227832045442305, (6.0, 12.0): 0, (6.0, 13.0): 0, (7.0, 1.0): 0, (7.0, 3.0): 0.8808553020503654, (7.0, 4.0): 0.25, (7.0, 5.0): 0.8808553020503654, (7.0, 6.0): 0.5, (7.0, 8.0): 0, (7.0, 9.0): 0, (7.0, 10.0): 0, (7.0, 11.0): 0, (7.0, 2.0): 0.25, (7.0, 12.0): 0, (7.0, 13.0): 0, (8.0, 1.0): 0, (8.0, 3.0): 0.25, (8.0, 4.0): 0.2, (8.0, 5.0): 0.25, (8.0, 6.0): 0.9186875586089954, (8.0, 7.0): 0.5, (8.0, 9.0): 0.25, (8.0, 10.0): 0.25, (8.0, 11.0): 0.3333333333333333, (8.0, 2.0): 0.2, (8.0, 12.0): 0.5, (8.0, 13.0): 0, (9.0, 1.0): 0, (9.0, 3.0): 0.2, (9.0, 4.0): 0.16666666666666666, (9.0, 5.0): 0.2, (9.0, 6.0): 0.25, (9.0, 7.0): 0.9041532667211214, (9.0, 8.0): 0.5, (9.0, 10.0): 0.7666883850009094, (9.0, 11.0): 0.25, (9.0, 2.0): 0.5, (9.0, 12.0): 0.9041532667211214, (9.0, 13.0): 0, (10.0, 1.0): 0, (10.0, 3.0): 0, (10.0, 4.0): 0, (10.0, 5.0): 0, (10.0, 6.0): 0, (10.0, 7.0): 0, (10.0, 8.0): 0, (10.0, 9.0): 0, (10.0, 11.0): 0, (10.0, 2.0): 0.5, (10.0, 12.0): 0, (10.0, 13.0): 0, (11.0, 1.0): 0, (11.0, 3.0): 0.16666666666666666, (11.0, 4.0): 0.14285714285714285, (11.0, 5.0): 0.16666666666666666, (11.0, 6.0): 0.2, (11.0, 7.0): 0.25, (11.0, 8.0): 0.3333333333333333, (11.0, 9.0): 0.5, (11.0, 10.0): 0.5, (11.0, 2.0): 0.7593282500242502, (11.0, 12.0): 0.25, (11.0, 13.0): 0, (2.0, 1.0): 0, (2.0, 3.0): 0, (2.0, 4.0): 0, (2.0, 5.0): 0, (2.0, 6.0): 0, (2.0, 7.0): 0, (2.0, 8.0): 0, (2.0, 9.0): 0, (2.0, 10.0): 0, (2.0, 11.0): 0, (2.0, 12.0): 0, (2.0, 13.0): 0, (12.0, 1.0): 0, (12.0, 3.0): 0.14285714285714285, (12.0, 4.0): 0.125, (12.0, 5.0): 0.14285714285714285, (12.0, 6.0): 0.16666666666666666, (12.0, 7.0): 0.8250727398856298, (12.0, 8.0): 0.25, (12.0, 9.0): 0.8679384451837155, (12.0, 10.0): 0.8679384451837155, (12.0, 11.0): 0.5, (12.0, 2.0): 0.25, (12.0, 13.0): 0, (13.0, 1.0): 0, (13.0, 3.0): 0.125, (13.0, 4.0): 0.1111111111111111, (13.0, 5.0): 0.125, (13.0, 6.0): 0.14285714285714285, (13.0, 7.0): 0.16666666666666666, (13.0, 8.0): 0.2, (13.0, 9.0): 0.25, (13.0, 10.0): 0.25, (13.0, 11.0): 0.3333333333333333, (13.0, 2.0): 0.2, (13.0, 12.0): 0.5}
        # similarity_matrix = {(1.0, 3.0): 0, (1.0, 4.0): 0, (1.0, 5.0): 0, (1.0, 6.0): 0, (1.0, 7.0): 0, (1.0, 8.0): 0, (1.0, 9.0): 0, (1.0, 10.0): 0, (1.0, 11.0): 0, (1.0, 2.0): 0, (1.0, 12.0): 0, (1.0, 13.0): 0, (3.0, 1.0): 0, (3.0, 4.0): 0.5, (3.0, 5.0): 0.5, (3.0, 6.0): 0.3333333333333333, (3.0, 7.0): 0.9999999740537899, (3.0, 8.0): 0.9983795616287776, (3.0, 9.0): 0.9999999999993843, (3.0, 10.0): 0.9999994915266016, (3.0, 11.0): 0.9983795616287776, (3.0, 2.0): 0.5, (3.0, 12.0): 0.999999989909807, (3.0, 13.0): 0.16666666666666666, (4.0, 1.0): 0, (4.0, 3.0): 0.5, (4.0, 5.0): 0.5, (4.0, 6.0): 0.3333333333333333, (4.0, 7.0): 0.9999999740537899, (4.0, 8.0): 0.9989191239214943, (4.0, 9.0): 0.5, (4.0, 10.0): 0.9999996186449027, (4.0, 11.0): 0.9989191239214943, (4.0, 2.0): 0.9999996378342734, (4.0, 12.0): 0.9999999924323554, (4.0, 13.0): 0.2, (5.0, 1.0): 0, (5.0, 3.0): 0.5, (5.0, 4.0): 0.5, (5.0, 6.0): 0.5, (5.0, 7.0): 0.9999999827025264, (5.0, 8.0): 0.9983795616287776, (5.0, 9.0): 0.9999999999993843, (5.0, 10.0): 0.9999994915266016, (5.0, 11.0): 0.9983795616287776, (5.0, 2.0): 0.9999996378342734, (5.0, 12.0): 0.999999989909807, (5.0, 13.0): 0.16666666666666666, (6.0, 1.0): 0, (6.0, 3.0): 0.3333333333333333, (6.0, 4.0): 0.3333333333333333, (6.0, 5.0): 0.5, (6.0, 7.0): 0.5, (6.0, 8.0): 0.3333333333333333, (6.0, 9.0): 0.9999999999990765, (6.0, 10.0): 0.9999993644083328, (6.0, 11.0): 0.9978405819063494, (6.0, 2.0): 0.9999994567515085, (6.0, 12.0): 0.9999999924323554, (6.0, 13.0): 0.2, (7.0, 1.0): 0, (7.0, 3.0): 0.25, (7.0, 4.0): 0.25, (7.0, 5.0): 0.3333333333333333, (7.0, 6.0): 0.5, (7.0, 8.0): 0.5, (7.0, 9.0): 0.3333333333333333, (7.0, 10.0): 0.9999994915266016, (7.0, 11.0): 0.9983795616287776, (7.0, 2.0): 0.9999992756688094, (7.0, 12.0): 0.9999999949549034, (7.0, 13.0): 0.25, (8.0, 1.0): 0, (8.0, 3.0): 0.9999999999990765, (8.0, 4.0): 0.9989191239214943, (8.0, 5.0): 0.25, (8.0, 6.0): 0.9999999999991791, (8.0, 7.0): 0.5, (8.0, 9.0): 0.5, (8.0, 10.0): 0.9999996186449027, (8.0, 11.0): 0.9989191239214943, (8.0, 2.0): 0.9999992756688094, (8.0, 12.0): 0.5, (8.0, 13.0): 0.3333333333333333, (9.0, 1.0): 0, (9.0, 3.0): 0.9999999999993843, (9.0, 4.0): 0.5, (9.0, 5.0): 0.9999999999993843, (9.0, 6.0): 0.25, (9.0, 7.0): 0.3333333333333333, (9.0, 8.0): 0.5, (9.0, 10.0): 0.9999997457632361, (9.0, 11.0): 0.5, (9.0, 2.0): 0.9999994567515085, (9.0, 12.0): 0.9999999949549034, (9.0, 13.0): 0.25, (10.0, 1.0): 0, (10.0, 3.0): 0.9999999999987685, (10.0, 4.0): 0.9983795616287776, (10.0, 5.0): 0.9999999999987685, (10.0, 6.0): 0.16666666666666666, (10.0, 7.0): 0.2, (10.0, 8.0): 0.9983795616287776, (10.0, 9.0): 0.3333333333333333, (10.0, 11.0): 0.5, (10.0, 2.0): 0.9999990945861756, (10.0, 12.0): 0.999999989909807, (10.0, 13.0): 0.16666666666666666, (11.0, 1.0): 0, (11.0, 3.0): 0.9999999999990765, (11.0, 4.0): 0.9989191239214943, (11.0, 5.0): 0.9999999999990765, (11.0, 6.0): 0.2, (11.0, 7.0): 0.25, (11.0, 8.0): 0.9989191239214943, (11.0, 9.0): 0.5, (11.0, 10.0): 0.5, (11.0, 2.0): 0.9999992756688094, (11.0, 12.0): 0.9999999924323554, (11.0, 13.0): 0.2, (2.0, 1.0): 0, (2.0, 3.0): 0.5, (2.0, 4.0): 0.3333333333333333, (2.0, 5.0): 0.3333333333333333, (2.0, 6.0): 0.25, (2.0, 7.0): 0.9999999654050536, (2.0, 8.0): 0.9978405819063494, (2.0, 9.0): 0.9999999999990765, (2.0, 10.0): 0.9999993644083328, (2.0, 11.0): 0.9978405819063494, (2.0, 12.0): 0.9999999873872588, (2.0, 13.0): 0.9993739447174214, (12.0, 1.0): 0, (12.0, 3.0): 0.9999999999987685, (12.0, 4.0): 0.9983795616287776, (12.0, 5.0): 0.2, (12.0, 6.0): 0.9999999999987685, (12.0, 7.0): 0.3333333333333333, (12.0, 8.0): 0.5, (12.0, 9.0): 0.3333333333333333, (12.0, 10.0): 0.9999994915266016, (12.0, 11.0): 0.9983795616287776, (12.0, 2.0): 0.9999990945861756, (12.0, 13.0): 0.5, (13.0, 1.0): 0, (13.0, 3.0): 0.9999999999984608, (13.0, 4.0): 0.9978405819063494, (13.0, 5.0): 0.16666666666666666, (13.0, 6.0): 0.9999999999983582, (13.0, 7.0): 0.25, (13.0, 8.0): 0.3333333333333333, (13.0, 9.0): 0.25, (13.0, 10.0): 0.9999993644083328, (13.0, 11.0): 0.9978405819063494, (13.0, 2.0): 0.9999989135036075, (13.0, 12.0): 0.5}
        recommendations = []

        print("Courses:", self.courses)
        print("Pretest Results:", self.pretest_results)
        
        course_ids = sorted([course['id'] for course in self.courses])
        print("Sorted Course IDs:", course_ids)
        
        course_points_and_similarity_scores = np.zeros(len(course_ids))
        for i, course_i in enumerate(course_ids):
            total_score = 0
            total_weight = 0
            
            for j, course_j in enumerate(course_ids):
                score = self.pretest_results[j]['score']
                # Handling self-similarity
                if i == j:
                    total_score += score
                    total_weight += 1
                else:
                    # print(similarity_matrix.get((5.0, 2.0), 0))
                    similarity = similarity_matrix.get((course_j, course_i), 0)
                    total_score += (score * similarity)
                    total_weight += similarity
            
            # Avoid division by zero
            print(course_i)
            print("Total Score:", total_score)
            print("Total Weight:", total_weight)
            if total_weight > 0:
                course_points_and_similarity_scores[i] = total_score / total_weight
            else:
                course_points_and_similarity_scores[i] = score  # Just use the direct score if no weights

            threshold = self.pass_threshold if total_weight < self.pass_threshold else 5
            if (course_points_and_similarity_scores[i] < threshold):
                recommendations.append(int(course_ids[i]))
        
        print("Final Course Points and Similarity Scores:", course_points_and_similarity_scores)
        print("Recommendations:", recommendations)
        
        return recommendations


# Assuming you have the pretest results saved in a JSON file at the following path:
if __name__ == "__main__":
    pretest_results_path = './data/pretestResults.json'
    pretest_results_path_2 = './data/pretestResults2.json'

    # Using the class to get recommendations
    recommender = SemanticSimilarityRecommender(pretest_results_path)
    learning_path_recommendations = recommender.get_recommended_courses()
    print(learning_path_recommendations)