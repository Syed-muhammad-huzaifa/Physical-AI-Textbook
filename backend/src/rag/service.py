"""
Service orchestration for the RAG (Retrieval Augmented Generation) system.

This module provides the business logic for ingesting and retrieving documents.
"""
from typing import Dict, Any, List
import logging
from .loader import load_markdown_files
from .chunking import chunk_document_text
from .embeddings import embedding_model
from .qdrant import qdrant_service
from .config import rag_config
import time


class RAGService:
    """
    Main service class for RAG operations
    """
    def __init__(self):
        self.qdrant_service = qdrant_service
        self.logger = logging.getLogger(__name__)

    def ingest_dir(self, wipe_collection: bool = False) -> Dict[str, int]:
        """
        Ingest all markdown files from the configured directory.

        Args:
            wipe_collection: If True, clear the collection before ingesting

        Returns:
            Dictionary with files_ingested and chunks_added counts
        """
        start_time = time.time()
        self.logger.info(f"Starting ingestion process, wipe_collection={wipe_collection}")

        # Clear collection if requested
        if wipe_collection:
            self.logger.info("Clearing collection as requested")
            self.qdrant_service.clear_collection()

        # Load markdown files
        self.logger.info(f"Loading markdown files from {rag_config.DOCS_DIR}")
        markdown_files = load_markdown_files()
        files_ingested = len(markdown_files)
        self.logger.info(f"Loaded {files_ingested} markdown files")

        if files_ingested == 0:
            self.logger.info("No markdown files found; exiting ingestion early")
            return {"files_ingested": 0, "chunks_added": 0}

        batch_size = rag_config.EMBEDDING_BATCH_SIZE
        self.logger.info(f"Processing embeddings in batches of {batch_size}")

        chunks_added = 0
        vector_size = None
        batch = []

        def process_batch(batch_items: List[Dict[str, Any]]) -> None:
            """Embed and upsert the current batch to avoid memory spikes."""
            nonlocal chunks_added, vector_size

            if not batch_items:
                return

            texts = [item["text"] for item in batch_items]
            embeddings = embedding_model.embed_documents(texts)

            if not embeddings:
                self.logger.warning("No embeddings generated for current batch; skipping upsert")
                return

            if vector_size is None:
                vector_size = len(embeddings[0])
                self.qdrant_service.ensure_collection(vector_size)

            chunks_for_upsert = []
            for item, embedding in zip(batch_items, embeddings):
                point_id = self.qdrant_service.get_point_id(item["doc_id"], item["chunk_id"])

                chunks_for_upsert.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": item["payload"]
                })

            self.qdrant_service.upsert_chunks(chunks_for_upsert)
            chunks_added += len(chunks_for_upsert)
            self.logger.debug(f"Upserted batch of {len(chunks_for_upsert)} chunks (total {chunks_added})")

        # Process each file
        for i, (doc_id, source, content) in enumerate(markdown_files):
            self.logger.debug(f"Processing file {i+1}/{files_ingested}: {doc_id}")

            # Chunk the document
            chunks = chunk_document_text(doc_id, content)
            self.logger.debug(f"Created {len(chunks)} chunks for {doc_id}")

            # Process each chunk
            for chunk in chunks:
                chunk_id = chunk["chunk_id"]
                text = chunk["text"]

                # Create payload
                payload = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": source,
                    "text": text
                }

                # Add to batch for embedding
                batch.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "payload": payload
                })

                if len(batch) >= batch_size:
                    process_batch(batch)
                    batch = []

        # Process any remaining chunks
        process_batch(batch)

        duration = time.time() - start_time
        self.logger.info(f"Ingestion completed: {files_ingested} files, {chunks_added} chunks in {duration:.2f}s")

        return {
            "files_ingested": files_ingested,
            "chunks_added": chunks_added
        }

    def retrieve(self, query: str, top_k: int = 6, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a given query.

        Args:
            query: The search query
            top_k: Number of results to return
            filters: Optional filters (e.g., {"doc_id": "some-doc.md"})

        Returns:
            List of dictionaries containing score, doc_id, chunk_id, source, text
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        self.logger.info(f"Starting retrieval for query (top_k={top_k}, filters={filters})")

        try:
            # Embed the query
            query_vector = embedding_model.embed_query(query)

            # Determine filter_doc_id from filters if present
            filter_doc_id = None
            if filters and "doc_id" in filters:
                filter_doc_id = filters["doc_id"]
                self.logger.debug(f"Applying filter for doc_id: {filter_doc_id}")

            # Search in Qdrant
            results = self.qdrant_service.search(
                query_vector=query_vector,
                top_k=top_k,
                filter_doc_id=filter_doc_id
            )

            self.logger.info(f"Retrieved {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Retrieval failed; returning empty results: {str(e)}")
            return []


# Global instance for reuse
rag_service = RAGService()
