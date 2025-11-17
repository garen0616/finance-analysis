import os
import logging
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class GraphClient:
    def __init__(self):
        self.enabled = os.environ.get("ENABLE_GRAPH", "true").lower() == "true"
        if not self.enabled:
            self.driver = None
            return
        uri = os.environ.get("NEO4J_URI")
        user = os.environ.get("NEO4J_USERNAME")
        pwd = os.environ.get("NEO4J_PASSWORD")
        if not (uri and user and pwd):
            self.enabled = False
            self.driver = None
            logger.warning("Graph disabled: missing credentials")
            return
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd), database=os.environ.get("NEO4J_DATABASE") or None)

    def run(self, query, params=None):
        if not self.enabled or not self.driver:
            return None
        with self.driver.session() as session:
            result = session.run(query, params or {})
            try:
                return list(result)
            finally:
                try:
                    result.consume()
                except Exception:
                    pass

    def close(self):
        if self.driver:
            self.driver.close()
