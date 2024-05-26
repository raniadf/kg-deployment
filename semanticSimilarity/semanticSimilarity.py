import sys
import os
import math
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from databaseHandler import DatabaseHandler
from .informationContent import InformationContentCalculator, TextProcessor

class SemanticSimilarity:
    def __init__(self):
        self.db = DatabaseHandler()
        with self.db.driver.session() as session:
            self.courses = session.execute_read(self.db.fetch_courses)
        self.ic_calculator = InformationContentCalculator("./data/text")
    
    def create_course_tokens(self):
        textProcessor = TextProcessor()
        course_tokens = {}
        for course in self.courses:
            preprocessed_text = textProcessor.preprocess_text(course['name'])
            tokens = textProcessor.tokenize_and_filter(preprocessed_text)
            lemmatized_tokens = textProcessor.lemmatize_tokens(tokens)
            combined_tokens = tokens + lemmatized_tokens
            combined_tokens = list(dict.fromkeys(combined_tokens))
            course_tokens[course['id']] = combined_tokens
        return course_tokens

    def calculate_sim_wpath(self, id1, id2):
        # Step 1: Fetch the LCS and its IC
        print("ID1: ", id1, "ID2: ", id2)
        try: 
            with self.db.driver.session() as session:
                lcs_id = session.execute_read(self.db.fetch_lcs, id1, id2)
                print("LCS ID: ", lcs_id)
            if lcs_id is not None:
                lcs_course = self.course_tokens.get(lcs_id, [])
                ic_lcs = sum(self.ic_calculator.calculate_ic(word) for word in lcs_course)
                print("IC LCS: ", ic_lcs)
            else:
                ic_lcs = 0 
                lcs_id = None
        except Exception as e:
            print(f"An error occurred: {e}")
            print("ID1: ", id1, "ID2: ", id2)
            ic_lcs = 0
            lcs_id = None

        # Step 2: Fetch the shortest path length and its nodes
        with self.db.driver.session() as session:
            path_length, path_nodes = session.execute_read(self.db.fetch_shortest_path_length, id1, id2)

        # Check if the LCS is in the shortest path
        if lcs_id and lcs_id in path_nodes:
            k = 1/(1+ic_lcs) if ic_lcs > 0 else 0.1
        else:
            k = 1 

        # Step 3: Calculate sim_wpath
        if path_length is None:
            return 0
        sim_wpath = 1 / (1 + path_length * k)
        
        return sim_wpath

    def calculate_semantic_similarity(self):
        self.course_tokens = self.create_course_tokens()
        print(self.course_tokens)
        similarity_matrix = {}
        for i in range(0, len(self.courses)):
            for j in range(0, len(self.courses)):
                print("j", j)
                if i == j:
                    continue
                id1 = self.courses[i]['id']
                id2 = self.courses[j]['id']
                similarity_matrix[(id1, id2)] = self.calculate_sim_wpath(id1, id2)
            print(i)
        print(similarity_matrix)
        return similarity_matrix

    def print_similarity_scores(self):
        similarity_matrix = self.calculate_semantic_similarity()
        for i in range(1, len(self.courses)+1):
            for j in range(1, len(self.courses)+1):
                if i == j:
                    print(1.00, end=" ")
                else:
                    print(round(similarity_matrix[(i, j)],2), end=" ")
            print()

# Usage
if __name__ == "__main__":
    similarity_instance = SemanticSimilarity()
    similarity_instance.print_similarity_scores()
    similarity_instance.db.close()