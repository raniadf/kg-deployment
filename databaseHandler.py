from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv() 

class DatabaseHandler:
    def __init__(self):
        self.driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))

    def verify_connectivity(self):
        self.driver.verify_connectivity()

    def session(self):
        return self.driver.session()

    def close(self):
        self.driver.close()

    def fetch_courses(self, tx):
        query = (
            "MATCH (c:Course) "
            "RETURN c.id AS id, c.name AS name"
        )
        result = tx.run(query)
        return [{"id": record["id"], "name": record["name"]} for record in result]
    
    def fetch_relations(self, tx):
        query = (
            "MATCH p=(c1:Course)-[r]->(c2:Course)"
            "RETURN c1.id, c1.name, c2.id, c2.name, type(r) AS relation"
        )
        result = tx.run(query)
        return [{"source_id": record["c1.id"], "source_name": record["c1.name"], "target_id": record["c2.id"], "target_name": record["c2.name"], "relation": record["relation"]} for record in result]
    
    def fetch_shortest_path_length(self, tx, id1, id2):
        query = (
            "MATCH p=shortestPath((c1:Course {id: $id1})<-[*..13]-(c2:Course {id: $id2})) "
            "WHERE NONE (n IN nodes(p) WHERE n.id = 1) "
            "RETURN p, length(p) AS length"
        )
        result = tx.run(query, id1=id1, id2=id2)
        # Fetch the record if available
        record = result.single()
        if record:
            path = record['p']
            length = record['length']
            nodes = [node.id for node in path.nodes]
            return length, nodes
        else:
            print(f"No path found between courses {id1} and {id2}")
            return None, []
    
    def fetch_lcs(self, tx, id1, id2):
        query = (
            "MATCH (c1:Course {id: $id1})-[]-(common:Course)-[]-(c2:Course {id: $id2}) "
            "WHERE c1 <> c2 AND c1 <> common AND c2 <> common "
            "RETURN common.id AS lcs_id "
            "LIMIT 1"
        )
        result = tx.run(query, id1=id1, id2=id2)
        return result.single()["lcs_id"]
    
    def create_course(self, tx, id, name):
        query = (
            "CREATE (c:Course {id: $id, name: $name})"
        )
        tx.run(query, id=id, name=name)
    
    def create_prerequisite_relation(self, tx, id1, id2): 
        query = (
            "MATCH (c1:Course {id: $id1}), (c2:Course {id: $id2}) "
            "CREATE (c1)-[:Prerequisites]->(c2)"
        )
        tx.run(query, id1=id1, id2=id2)

    def create_enables_relation(self, tx, id1, id2):
        query = (
            "MATCH (c1:Course {id: $id1}), (c2:Course {id: $id2}) "
            "CREATE (c1)-[:Enables]->(c2)"
        )
        tx.run(query, id1=id1, id2=id2)

    def create_similar_relation(self, tx, id1, id2):
        query = (
            "MATCH (c1:Course {id: $id1}), (c2:Course {id: $id2}) "
            "CREATE (c1)-[:Similar]->(c2)"
        )
        tx.run(query, id1=id1, id2=id2)

    def create_properties_node(self, tx, id, propertiesName, propertiesValue):
        query = (
            "MATCH (c:Course {id: $id}) "
            "SET c.{propertiesName: $propertiesName} = {propertiesValue: $propertiesValue}"
        )
        tx.run(query, id=id, propertiesName=propertiesName, propertiesValue=propertiesValue)

    def create_properties_relation(self, tx, id1, id2, propertiesName, propertiesValue):
        query = (
            "MATCH (c1:Course {id: $id1})-[r]-(c2:Course {id: $id2}) "
            "SET r.{propertiesName: $propertiesName} = {propertiesValue: $propertiesValue}"
        )
        tx.run(query, id1=id1, id2=id2, propertiesName=propertiesName, propertiesValue=propertiesValue)

    def delete_course(self, tx, id):
        query = (
            "MATCH (c:Course {id: $id}) "
            "DETACH DELETE c"
        )
        tx.run(query, id=id)

    def delete_relation(self, tx, id1, id2):
        query = (
            "MATCH (c1:Course {id: $id1})-[r]-(c2:Course {id: $id2}) "
            "DELETE r"
        )
        tx.run(query, id1=id1, id2=id2)

    def execute_write(self, query, *args):
        with self.session() as session:
            return session.write_transaction(query, *args)
        
    def execute_read(self, query, *args):
        with self.session() as session:
            return session.execute_read(query, *args)

if __name__ == "__main__":
    greeter = DatabaseHandler()
    greeter.verify_connectivity()
    with greeter.session() as session:
        print(session.execute_read(greeter.fetch_courses))
        print(session.execute_read(greeter.fetch_shortest_path_length, 11, 3))
        print(session.execute_read(greeter.fetch_lcs, 11, 3))
    greeter.close()
