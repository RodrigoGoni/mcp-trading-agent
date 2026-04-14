"""
src/memory/qdrant_store.py
Almacena y recupera decisiones del agente usando Qdrant como vector store.
Usa sentence-transformers para los embeddings (ya instalado en el venv).
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
    Almacena decisiones del agente en Qdrant para análisis posterior
    y para permitir que el agente consulte decisiones históricas similares.
    """

    VECTOR_DIM = 384  # all-MiniLM-L6-v2 produce vectores de 384 dims

    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self.encoder = SentenceTransformer(settings.embedding_model)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Crea la colección si no existe."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.VECTOR_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Colección '{self.collection}' creada en Qdrant.")
        else:
            logger.debug(f"Colección '{self.collection}' ya existe.")

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
        Guarda la decisión del agente para una fecha dada.
        Retorna el ID del punto guardado.
        """
        point_id = str(uuid.uuid4())
        text_to_embed = (
            f"Fecha: {date}. "
            f"Portfolio: ${portfolio_value:,.2f}. "
            f"Trades: {trades_executed}. "
            f"Resumen: {agent_summary}"
        )
        vector = self._embed(text_to_embed)
        payload = {
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
        logger.debug(f"Decisión guardada en Qdrant: {point_id} ({date})")
        return point_id

    def get_similar_decisions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Busca decisiones históricas similares a la query dada.
        Útil para que el agente consulte contexto de situaciones pasadas.
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

    def get_all_decisions(self) -> List[Dict[str, Any]]:
        """Retorna todas las decisiones almacenadas, ordenadas por fecha."""
        results, _ = self.client.scroll(
            collection_name=self.collection,
            limit=10_000,
            with_payload=True,
        )
        decisions = [r.payload for r in results]
        return sorted(decisions, key=lambda x: x.get("date", ""))
