"""
Qdrant Vector Database Storage for Financial Reports
===================================================

Store financial report chunks as embeddings in Qdrant for semantic search.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from core_services.data_ingestion.text_processor import FinancialReportChunker
from core_services.utils.logger_utils import logger


class QdrantFinancialReportStorage:
    """Store financial reports in Qdrant for semantic search"""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "financial_reports_embeddings",
        embedding_model: str = "keepitreal/vietnamese-sbert"
    ):
        """
        Initialize Qdrant storage

        Args:
            qdrant_url: Qdrant server URL
            collection_name: Collection name for financial reports
            embedding_model: Sentence transformer model for Vietnamese
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(embedding_model)
        self.chunker = FinancialReportChunker()

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if not exists"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,  # Vietnamese SBERT dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info(
                    f"Created Qdrant collection: {self.collection_name}"
                )
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")

    def store_financial_report(
        self,
        financial_data: Dict[str, Any]
    ) -> bool:
        """
        Chunk, embed, and store financial report

        Args:
            financial_data: Financial report data from crawler

        Returns:
            True if successful
        """
        try:
            # Create chunks
            chunks = self.chunker.create_chunks(financial_data)

            if not chunks:
                logger.warning(
                    f"No chunks created for {financial_data.get('symbol')}"
                )
                return False

            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.encoder.encode(texts, show_progress_bar=False)

            # Create points
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())

                payload = {
                    "symbol": financial_data['symbol'],
                    "fiscal_year": financial_data['fiscal_year'],
                    "fiscal_quarter": financial_data.get('fiscal_quarter'),
                    "period_type": financial_data['period_type'],
                    "chunk_type": chunk['chunk_type'],
                    "text": chunk['text'],
                    "created_at": datetime.now().isoformat(),

                    # Financial metrics for filtering
                    "revenue": financial_data.get('revenue'),
                    "net_profit": financial_data.get('net_profit'),
                    "eps": financial_data.get('eps'),
                    "roe": financial_data.get('roe'),
                    "roa": financial_data.get('roa'),
                    "total_assets": financial_data.get('total_assets'),
                    "shareholders_equity": financial_data.get(
                        'shareholders_equity'
                    ),
                }

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                ))

            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(
                f"Stored {len(points)} chunks for {financial_data['symbol']} "
                f"{financial_data['fiscal_year']}"
                f"Q{financial_data.get('fiscal_quarter', 'Y')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store in Qdrant: {e}")
            return False

    def search_similar_reports(
        self,
        query: str,
        symbol: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search for similar financial reports

        Args:
            query: Natural language query in Vietnamese
            symbol: Optional stock symbol filter
            year: Optional year filter
            limit: Number of results

        Returns:
            List of matching financial report chunks
        """
        try:
            # Encode query
            query_vector = self.encoder.encode(query).tolist()

            # Build filter
            filter_conditions = None
            must_conditions = []

            if symbol:
                must_conditions.append({
                    "key": "symbol",
                    "match": {"value": symbol.upper()}
                })

            if year:
                must_conditions.append({
                    "key": "fiscal_year",
                    "match": {"value": year}
                })

            if must_conditions:
                filter_conditions = {"must": must_conditions}

            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filter_conditions,
                limit=limit
            )

            return [
                {
                    "score": hit.score,
                    "symbol": hit.payload.get('symbol'),
                    "year": hit.payload.get('fiscal_year'),
                    "quarter": hit.payload.get('fiscal_quarter'),
                    "text": hit.payload.get('text'),
                    "chunk_type": hit.payload.get('chunk_type'),
                    "revenue": hit.payload.get('revenue'),
                    "net_profit": hit.payload.get('net_profit'),
                    "eps": hit.payload.get('eps'),
                    "roe": hit.payload.get('roe'),
                }
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def delete_financial_report(
        self,
        symbol: str,
        year: int,
        quarter: Optional[int] = None
    ) -> bool:
        """
        Delete financial report from Qdrant

        Args:
            symbol: Stock symbol
            year: Fiscal year
            quarter: Optional fiscal quarter

        Returns:
            True if successful
        """
        try:
            # Build filter
            must_conditions = [
                {"key": "symbol", "match": {"value": symbol.upper()}},
                {"key": "fiscal_year", "match": {"value": year}}
            ]

            if quarter is not None:
                must_conditions.append({
                    "key": "fiscal_quarter",
                    "match": {"value": quarter}
                })

            filter_conditions = {"must": must_conditions}

            # Delete points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_conditions
            )

            logger.info(
                f"Deleted financial report chunks for {symbol} "
                f"{year}Q{quarter if quarter else 'Y'}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
