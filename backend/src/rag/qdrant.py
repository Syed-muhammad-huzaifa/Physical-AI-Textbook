"""
Qdrant client wrapper for the RAG (Retrieval Augmented Generation) system.

This module provides functionality to interact with Qdrant vector database.
"""
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from .config import rag_config
import hashlib
import logging


class QdrantService:
    """
    Service class for interacting with Qdrant vector database
    """
    def __init__(self):
        # Initialize Qdrant client with configuration
        self.client = QdrantClient(
            url=rag_config.QDRANT_URL,
            api_key=rag_config.QDRANT_API_KEY,
            # Cloud endpoints often block gRPC; prefer HTTP to avoid connection refusals
            prefer_grpc=False
        )
        self.collection_name = rag_config.QDRANT_COLLECTION
        self.logger = logging.getLogger(__name__)

    def ensure_collection(self, vector_size: int) -> None:
        """
        Ensure that the collection exists with the specified vector size.

        Args:
            vector_size: Size of the embedding vectors
        """
        try:
            # Try to get collection info to check if it exists
            self.client.get_collection(self.collection_name)
            self.logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            self.logger.info(f"Collection {self.collection_name} does not exist, creating it...")
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            self.logger.info(f"Collection {self.collection_name} created successfully")

    def upsert_chunks(self, chunks_with_vectors_and_payload: List[Dict[str, Any]]) -> None:
        """
        Upsert chunks with their vectors and payload to Qdrant.

        Args:
            chunks_with_vectors_and_payload: List of dictionaries containing:
                - vector: embedding vector
                - payload: metadata including doc_id, chunk_id, source, text
                - id: point ID for the chunk
        """
        if not chunks_with_vectors_and_payload:
            self.logger.info("No chunks to upsert")
            return

        # Prepare points for upsert
        points = []
        for item in chunks_with_vectors_and_payload:
            point_id = item.get('id')
            vector = item.get('vector')
            payload = item.get('payload', {})

            points.append(models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        try:
            # Upsert the points to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            self.logger.info(f"Successfully upserted {len(points)} chunks to Qdrant")
        except Exception as e:
            self.logger.error(f"Failed to upsert chunks to Qdrant: {str(e)}")
            raise

    def search(self, query_vector: List[float], top_k: int = 6, filter_doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Qdrant.

        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            filter_doc_id: Optional document ID to filter results

        Returns:
            List of dictionaries containing score, doc_id, chunk_id, source, text
        """
        try:
            # Prepare filters if doc_id is specified
            filters = None
            if filter_doc_id:
                filters = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=filter_doc_id)
                        )
                    ]
                )

            # Perform search using query_points (new client API)
            search_response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=filters,
                limit=top_k
            )

            # Format results
            results = []
            for hit in search_response.points:
                results.append({
                    "score": hit.score,
                    "doc_id": hit.payload.get("doc_id"),
                    "chunk_id": hit.payload.get("chunk_id"),
                    "source": hit.payload.get("source"),
                    "text": hit.payload.get("text")
                })

            self.logger.info(f"Search completed, found {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise

    def get_point_id(self, doc_id: str, chunk_id: int) -> str:
        """
        Generate a stable point ID based on doc_id and chunk_id.

        Args:
            doc_id: Document ID
            chunk_id: Chunk ID within the document

        Returns:
            Stable point ID
        """
        # Create a hash of the doc_id and chunk_id to ensure stable IDs
        id_str = f"{doc_id}_{chunk_id}"
        return hashlib.md5(id_str.encode()).hexdigest()

    def clear_collection(self) -> None:
        """
        Clear all points from the collection.
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            self.logger.info(f"Collection {self.collection_name} cleared successfully")
        except Exception as e:
            self.logger.error(f"Failed to clear collection {self.collection_name}: {str(e)}")
            raise


# Global instance for reuse
qdrant_service = QdrantService()
