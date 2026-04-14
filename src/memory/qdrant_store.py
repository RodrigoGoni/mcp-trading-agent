"""
src/memory/qdrant_store.py
Stores and retrieves agent decisions using Qdrant as a vector store.
Uses sentence-transformers for embeddings (already installed in the venv).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)


class DecisionStore:
    """
    Stores agent decisions in Qdrant for later analysis
    and to allow the agent to query similar historical decisions.
    """

    VECTOR_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    def __init__(self, run_id: Optional[str] = None) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self.encoder = SentenceTransformer(settings.embedding_model)
        self.run_id = run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the collection if it does not exist."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection}' created in Qdrant.")
        else:
            logger.debug(f"Collection '{self.collection}' already exists.")

    def clear_collection(self) -> None:
        """Deletes and recreates the collection (removes ALL runs)."""
        self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{self.collection}' cleared.")

    def _embed(self, text: str) -> List[float]:
        return self.encoder.encode(text, normalize_embeddings=True).tolist()

    def save_decision(
        self,
        date: str,
        portfolio_value: float,
        trades_executed: List[Dict[str, Any]],
        agent_summary: str,
        portfolio_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Saves the agent's decision for a given date.
        Returns the ID of the saved point.
        """
        point_id = str(uuid.uuid4())
        text_to_embed = (
            f"Date: {date}. "
            f"Portfolio: ${portfolio_value:,.2f}. "
            f"Trades: {trades_executed}. "
            f"Summary: {agent_summary}"
        )
        vector = self._embed(text_to_embed)
        payload = {
            "run_id": self.run_id,
            "date": date,
            "portfolio_value": portfolio_value,
            "trades": trades_executed,
            "agent_summary": agent_summary,
            "portfolio_snapshot": portfolio_snapshot or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.debug(f"Decision saved in Qdrant: {point_id} ({date})")
        return point_id

    def get_similar_decisions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for historical decisions similar to the given query.
        Useful for the agent to consult context from past situations.
        """
        vector = self._embed(query)
        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "score": round(r.score, 4),
                "date": r.payload.get("date"),
                "portfolio_value": r.payload.get("portfolio_value"),
                "agent_summary": r.payload.get("agent_summary"),
                "trades": r.payload.get("trades"),
            }
            for r in results
        ]

    def get_all_decisions(self, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all stored decisions, sorted by date. Optionally filters by run_id."""
        scroll_filter = None
        if run_id:
            scroll_filter = Filter(
                must=[FieldCondition(key="run_id", match=MatchValue(value=run_id))]
            )
        results, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=scroll_filter,
            limit=10_000,
            with_payload=True,
        )
        decisions = [r.payload for r in results]
        return sorted(decisions, key=lambda x: x.get("date", ""))
