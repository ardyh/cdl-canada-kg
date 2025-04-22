from neo4j import GraphDatabase
import json
import datetime
from urllib.parse import urlparse

class RedditKnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="your_password", database="neo4j"):
        """Initialize connection to Neo4j database."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        """Close the driver connection."""
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships in the database."""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")

    def get_subreddit_from_permalink(self, permalink):
        """Extract subreddit from permalink."""
        parts = permalink.strip('/').split('/')
        if len(parts) > 1:
            return parts[1]
        return None

    def create_indices(self):
        """Create indices for better performance."""
        with self.driver.session(database=self.database) as session:
            # Create constraints (which automatically create indices)
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Action) REQUIRE a.action IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (o:Object) REQUIRE o.object IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Subreddit) REQUIRE s.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Narrative) REQUIRE n.narrative IS UNIQUE")

    def create_entity_node(self, tx, entity_name):
        """Create an Entity node."""
        query = """
        MERGE (e:Entity {name: $name})
        RETURN e
        """
        return tx.run(query, name=entity_name)

    def create_post_node(self, tx, post_id, post_data):
        """Create a Post node."""
        query = """
        MERGE (p:Post {id: $id})
        SET 
            p.text = $text,
            p.score = $score,
            p.created_utc = $created_utc,
            p.created_date = $created_date,
            p.permalink = $permalink
        RETURN p
        """
        return tx.run(
            query,
            id=post_id,
            text=post_data['text'],
            score=post_data.get('score', 0),
            created_utc=float(post_data.get('created_utc', 0)),
            created_date=datetime.datetime.fromtimestamp(
                float(post_data.get('created_utc', 0))).strftime('%Y-%m-%d'),
            permalink=post_data.get('permalink', '')
        )

    def create_subreddit_node(self, tx, subreddit_name):
        """Create a Subreddit node."""
        query = """
        MERGE (s:Subreddit {name: $name})
        RETURN s
        """
        return tx.run(query, name=subreddit_name)
    
    def create_action_node(self, tx, action, cluster):
        """Create an Action node."""
        query = """
        MERGE (a:Action {action: $action})
        SET a.cluster = $cluster
        RETURN a
        """
        return tx.run(query, action=action, cluster=cluster)
    
    def create_object_node(self, tx, obj):
        """Create an Object node."""
        query = """
        MERGE (o:Object {object: $object})
        RETURN o
        """
        return tx.run(query, object=obj)
    
    def create_narrative_node(self, tx, narr, cluster, sentiment):
        """Create a Narrative node."""
        query = """
        MERGE (n:Narrative {narrative: $narrative})
        SET 
            n.cluster = $cluster,
            n.sentiment = $sentiment
        RETURN n
        """
        return tx.run(query, narrative=narr, cluster=cluster, sentiment=sentiment)

    def create_relationships(self, tx, post_id, entity_name, subreddit_name, action, object_name, narrative, action_cluster, narrative_cluster, sentiment, created_utc):
        """Create relationships between nodes."""
        # The issue is that the relationships aren't being created properly
        # Let's make sure all parameters are valid before creating relationships
        if not all([post_id, entity_name, subreddit_name, action, object_name, narrative]):
            print(f"Missing required data for relationship: {post_id}, {entity_name}, {subreddit_name}, {action}, {object_name}, {narrative}")
            return
            
        query = """
        MATCH (p:Post {id: $post_id})
        MATCH (e:Entity {name: $entity_name})
        MATCH (a:Action {action: $action})
        MATCH (o:Object {object: $object_name})
        MATCH (s:Subreddit {name: $subreddit_name})
        MATCH (n:Narrative {narrative: $narrative})
        
        MERGE (p)-[m:MENTIONS]->(e)
        ON CREATE SET m.timestamp = $timestamp
        
        MERGE (p)-[pi:POSTED_IN]->(s)
        MERGE (e)-[do:DOES]->(a)
        MERGE (a)-[af:AFFECTS]->(o)
        MERGE (p)-[pn:CONTAINS_NARRATIVE]->(n)
        MERGE (e)-[en:PART_OF_NARRATIVE]->(n)
        MERGE (a)-[an:PART_OF_NARRATIVE]->(n)
        MERGE (o)-[on:PART_OF_NARRATIVE]->(n)
        """
        try:
            tx.run(
                query,
                post_id=post_id,
                entity_name=entity_name,
                subreddit_name=subreddit_name,
                action=action,
                object_name=object_name,
                narrative=narrative,
                timestamp=float(created_utc)
            )
        except Exception as e:
            print(f"Error creating relationships: {e}")
            print(f"Data: {post_id}, {entity_name}, {subreddit_name}, {action}, {object_name}, {narrative}")

    def create_knowledge_graph(self, ner_file):
        """Create knowledge graph from narrative data."""
        # Load JSON file
        with open(ner_file, 'r', encoding='utf-8') as f:
            narrative_data = json.load(f)

        # Create indices
        self.create_indices()
        
        # Process each post and its narratives
        with self.driver.session(database=self.database) as session:
            processed_count = 0
            for post_id, post_data in narrative_data.items():
                # Skip posts without political narratives
                if not post_data.get('political_narratives'):
                    continue
                
                # Create post node
                session.execute_write(self.create_post_node, post_id, post_data)
                
                # Create subreddit node
                subreddit_name = self.get_subreddit_from_permalink(post_data.get('permalink', ''))
                if subreddit_name:
                    session.execute_write(self.create_subreddit_node, subreddit_name)
                else:
                    # If we can't extract subreddit, use a default
                    subreddit_name = "unknown"
                    session.execute_write(self.create_subreddit_node, subreddit_name)
                
                # Process each narrative in the post
                for narrative in post_data.get('political_narratives', []):
                    entity_name = narrative.get('name', '')
                    action = narrative.get('action_or_role', '')
                    affected_entity = narrative.get('affected_entities', '')
                    narrative_text = narrative.get('narrative', '')
                    sentiment = narrative.get('sentiment', 'neutral')
                    action_cluster = narrative.get('action_cluster', '')
                    narrative_cluster = narrative.get('narrative_cluster', '')
                    
                    # Make sure we have all required data
                    if not entity_name:
                        entity_name = "Unknown Entity"
                    if not action:
                        action = "Unknown Action"
                    if not affected_entity:
                        affected_entity = "Unknown Object"
                    if not narrative_text:
                        narrative_text = f"{entity_name} {action} {affected_entity}"
                    
                    # Create entity node
                    session.execute_write(self.create_entity_node, entity_name)
                    
                    # Create action node
                    session.execute_write(self.create_action_node, action, action_cluster)
                    
                    # Create object node
                    session.execute_write(self.create_object_node, affected_entity)
                    
                    # Create narrative node
                    session.execute_write(self.create_narrative_node, narrative_text, narrative_cluster, sentiment)
                    
                    # Create relationships
                    session.execute_write(
                        self.create_relationships,
                        post_id,
                        entity_name,
                        subreddit_name,
                        action,
                        affected_entity,
                        narrative_text,
                        action_cluster,
                        narrative_cluster,
                        sentiment,
                        post_data.get('created_utc', 0)
                    )
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} posts with narratives")
            
            print(f"Finished processing {processed_count} posts with narratives")
            
            # Verify relationships were created
            relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            print(f"Created {relationship_count} relationships in the database")

    def get_example_queries(self):
        """Return example Cypher queries for exploring the knowledge graph."""
        return [
            """
            // Top entities by mention count
            MATCH (e:Entity)<-[m:MENTIONS]-()
            RETURN e.name, COUNT(m) as mentions
            ORDER BY mentions DESC
            LIMIT 10
            """,
            
            """
            // Top actions by entity
            MATCH (e:Entity)-[d:DOES]->(a:Action)
            RETURN e.name, a.action, COUNT(d) as count
            ORDER BY count DESC
            LIMIT 10
            """,
            
            """
            // Top narratives by sentiment
            MATCH (n:Narrative)
            RETURN n.sentiment, COUNT(n) as count
            ORDER BY count DESC
            """,
            
            """
            // Entity mentions over time
            MATCH (e:Entity)<-[m:MENTIONS]-(p:Post)
            RETURN e.name, p.created_date, COUNT(m) as mentions
            ORDER BY p.created_date
            """
        ]

def main():
    # Initialize knowledge graph
    kg = RedditKnowledgeGraph(
        uri="bolt://localhost:7687",
        user="cdl",
        password="cdlpassword",  # Replace with your password
        database="narratives"  # Specify target database
    )

    try:
        # Clear existing data
        kg.clear_database()

        # Create knowledge graph
        kg.create_knowledge_graph(
            ner_file="spacy_ner_output/narrative_v2_ent_obj_act_narr_deduplicated.json"
        )

    finally:
        # Close the driver connection
        kg.close()

if __name__ == "__main__":
    main()