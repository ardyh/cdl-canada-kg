from neo4j import GraphDatabase
import pandas as pd
import json
import os

class KnowledgeGraphQuerier:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="your_password", database="neo4j"):
        """Initialize connection to Neo4j database."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        """Close the driver connection."""
        self.driver.close()

    def run_query(self, query, parameters=None):
        """Run a Cypher query and return results."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record for record in result]
            
    def run_query_to_df(self, query, parameters=None):
        """Run a Cypher query and return results as a pandas DataFrame."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            # Convert result to DataFrame
            records = [record.values() for record in result]
            if not records:
                return pd.DataFrame()
            return pd.DataFrame(records, columns=result.keys())
    
    def get_politician_mentions(self, politician_name):
        """Get politician mentions across platforms."""
        query = """
        MATCH (p:Post)-[:MENTIONS]->(e:Entity)
        WHERE e.name =~ $name_pattern
        WITH
        e,
        COLLECT(DISTINCT p) AS posts
        WITH
        e.name AS name,
        SIZE([p IN posts WHERE p.platform = "X"]) AS count_x,
        SIZE([p IN posts WHERE p.platform = "Reddit"]) AS count_reddit,
        SIZE([p IN posts WHERE p.platform = "Bluesky"]) AS count_bluesky,
        SIZE(posts) AS post_counts
        RETURN
        name,
        post_counts,
        count_x,
        count_reddit,
        count_bluesky
        """
        # Create a case-insensitive regex pattern for the politician name
        name_pattern = f"(?i).*{politician_name}.*"
        return self.run_query_to_df(query, {"name_pattern": name_pattern})
    
    def get_politician_narratives(self, politician_name, limit=None):
        """Get narratives on a politician."""
        query = """
        // Narratives related to a single entity
        MATCH (e:Entity)-[:PART_OF_NARRATIVE]->(n:Narrative)
        MATCH (p:Post)-[:CONTAINS_NARRATIVE]->(n)
        WHERE e.name =~ $name_pattern
        WITH n.name AS narrative_cluster, n.cluster_id as narrative_cluster_id, COUNT(DISTINCT p) AS count_mentions, n.raw_names AS sample_narratives
        RETURN
        narrative_cluster_id,
        narrative_cluster,
        count_mentions,
        sample_narratives
        ORDER BY count_mentions DESC
        """
        name_pattern = f"(?i).*{politician_name}.*"
        if limit:
            query += f" LIMIT {limit}"
        return self.run_query_to_df(query, {"name_pattern": name_pattern})

    def get_co_mentions(self, politician_name, limit=None):
        """Get co-mentions of a politician."""
        query = """
        // Find entity pairs that frequently co-occur in the same posts
        MATCH (p:Post)-[:MENTIONS]->(e1:Entity)
        MATCH (p)-[:MENTIONS]->(e2:Entity)
        MATCH (p)-[:CONTAINS_NARRATIVE]->(n1:Narrative)
        MATCH (e1)-[:PART_OF_NARRATIVE]->(n1)
        WHERE e1.name < e2.name  // Avoid duplicates (A,B) and (B,A)
                AND e1.name =~ $name_pattern
        WITH e1.name AS entity1, e2.name AS entity2, COUNT(DISTINCT p) AS co_occurrences, COLLECT(DISTINCT n1.name) AS narratives1
        WITH 
            entity1,
            narratives1,
            entity2,
            co_occurrences
        WHERE co_occurrences >= 2  // Set minimum threshold of co-occurrences

        // Return results sorted by co-occurrence frequency
        RETURN 
            entity2, 
            co_occurrences,
            narratives1
        ORDER BY co_occurrences DESC
        """
        name_pattern = f"(?i).*{politician_name}.*"
        if limit:
            query += f" LIMIT {limit}"
        return self.run_query_to_df(query, {"name_pattern": name_pattern})
    
    def get_popular_narratives(self, cluster_id, limit=10):
        """Get most popular narratives."""
        query = """
        // Platform Counts of a certain narrative
        MATCH (p:Post)-[:CONTAINS_NARRATIVE]->(n:Narrative)
        WHERE n.cluster_id = $cluster_id
        WITH n, COLLECT(DISTINCT {id: p.id, text: p.text, platform: p.platform}) AS post_details
        WITH 
        n.cluster_id AS narrative_id,
        n.name AS narrative_cluster, 
        n.raw_names AS narratives,
        post_details
        WITH 
        narrative_id,
        narrative_cluster,
        narratives,
        SIZE([p IN post_details WHERE p.platform = "X"]) AS count_x,
        SIZE([p IN post_details WHERE p.platform = "Reddit"]) AS count_reddit,
        SIZE([p IN post_details WHERE p.platform = "Bluesky"]) AS count_bluesky,
        post_details
        RETURN
        narrative_id,
        narrative_cluster,
        narratives,
        count_x,
        count_reddit,
        count_bluesky,
        post_details
        """
        return self.run_query_to_df(query, {"cluster_id": cluster_id})
    
    def get_post_entities(self, post_id):
        """Get entities mentioned in a post."""
        query = """
        // (post.py) Find entities related to a post
        MATCH (p:Post)-[:MENTIONS]->(e1:Entity)
        MATCH (p)-[:MENTIONS]->(a:Action)
        MATCH (e1)-[:DOES]->(a)-[:AFFECTS]->(e2:Entity)
        MATCH (e1)-[:PART_OF_NARRATIVE]->(n:Narrative)
        MATCH (a)-[:PART_OF_NARRATIVE]->(n:Narrative)
        MATCH (e2)-[:PART_OF_NARRATIVE]->(n:Narrative)
        WHERE p.id = $post_id
        WITH DISTINCT e1.name AS actor, a.name AS action, e2.name AS target, n.name AS narrative
        RETURN actor, action, target, narrative
        ORDER BY actor ASC, action ASC
        """
        if post_id is str:
            formatted_post_id = f"\"{post_id}\""
        else:
            formatted_post_id = post_id
        return self.run_query_to_df(query, {"post_id": formatted_post_id})