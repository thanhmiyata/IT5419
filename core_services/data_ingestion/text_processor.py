"""
Text Processing for Financial Reports Embeddings
================================================

Chunk financial reports into semantic units for vector embedding
and storage in Qdrant vector database.
"""

from typing import Any, Dict, List


class FinancialReportChunker:
    """Chunk financial reports for embedding and semantic search"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize chunker

        Args:
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(
        self, financial_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from financial report

        Args:
            financial_data: Financial report data dict

        Returns:
            List of chunks with text and metadata
        """
        chunks = []

        # Chunk 1: Executive Summary
        summary_text = self._create_summary(financial_data)
        chunks.append({
            "text": summary_text,
            "chunk_type": "summary",
            "metadata": {
                "symbol": financial_data['symbol'],
                "year": financial_data['fiscal_year'],
                "quarter": financial_data.get('fiscal_quarter')
            }
        })

        # Chunk 2: Income Statement
        if any(
            financial_data.get(k) for k in ['revenue', 'net_profit', 'ebit']
        ):
            income_text = self._create_income_statement_text(financial_data)
            chunks.append({
                "text": income_text,
                "chunk_type": "income_statement",
                "metadata": financial_data.copy()
            })

        # Chunk 3: Balance Sheet
        if any(
            financial_data.get(k)
            for k in ['total_assets', 'shareholders_equity']
        ):
            balance_text = self._create_balance_sheet_text(financial_data)
            chunks.append({
                "text": balance_text,
                "chunk_type": "balance_sheet",
                "metadata": financial_data.copy()
            })

        # Chunk 4: Cash Flow
        if any(
            financial_data.get(k)
            for k in ['operating_cash_flow', 'free_cash_flow']
        ):
            cash_flow_text = self._create_cash_flow_text(financial_data)
            chunks.append({
                "text": cash_flow_text,
                "chunk_type": "cash_flow",
                "metadata": financial_data.copy()
            })

        # Chunk 5: Key Ratios Analysis
        if any(financial_data.get(k) for k in ['eps', 'roe', 'roa']):
            ratio_text = self._create_ratio_analysis(financial_data)
            chunks.append({
                "text": ratio_text,
                "chunk_type": "ratios",
                "metadata": financial_data.copy()
            })

        return chunks

    def _create_summary(self, data: Dict) -> str:
        """Create executive summary in Vietnamese"""
        symbol = data['symbol']
        year = data['fiscal_year']
        quarter = (
            f"Q{data['fiscal_quarter']}"
            if data.get('fiscal_quarter')
            else "Năm"
        )

        return f"""
Báo cáo tài chính {symbol} - {quarter} {year}

Tóm tắt hiệu suất kinh doanh:
- Doanh thu: {self._format_number(data.get('revenue'))} VND
- Lợi nhuận gộp: {self._format_number(data.get('gross_profit'))} VND
- Lợi nhuận ròng: {self._format_number(data.get('net_profit'))} VND
- EPS (Thu nhập/cổ phiếu): {data.get('eps', 'N/A')} VND
- ROE (Sinh lời/vốn): {data.get('roe', 'N/A')}%
- ROA (Sinh lời/tài sản): {data.get('roa', 'N/A')}%

Tình hình tài chính:
- Tổng tài sản: {self._format_number(data.get('total_assets'))} VND
- Vốn chủ sở hữu: {self._format_number(data.get('shareholders_equity'))} VND
- Nợ phải trả: {self._format_number(data.get('total_liabilities'))} VND
- Tỷ lệ nợ/vốn: {data.get('debt_to_equity', 'N/A')}
""".strip()

    def _create_income_statement_text(self, data: Dict) -> str:
        """Generate natural language income statement"""
        symbol = data['symbol']
        period = (
            f"quý {data['fiscal_quarter']} năm {data['fiscal_year']}"
            if data.get('fiscal_quarter')
            else f"năm {data['fiscal_year']}"
        )

        return f"""
Báo cáo kết quả kinh doanh của {symbol} trong {period}:

Doanh thu và lợi nhuận:
- Doanh thu thuần: {self._format_number(data.get('revenue'))} VND
- Giá vốn hàng bán: {self._format_number(data.get('cost_of_goods_sold'))} VND
- Lợi nhuận gộp: {self._format_number(data.get('gross_profit'))} VND
- Chi phí hoạt động: {self._format_number(data.get('operating_expenses'))} VND
- Lợi nhuận hoạt động: {self._format_number(data.get('operating_profit'))} VND

Lợi nhuận trước thuế và lãi vay:
- EBIT: {self._format_number(data.get('ebit'))} VND
- EBITDA: {self._format_number(data.get('ebitda'))} VND

Kết quả cuối cùng:
- Chi phí lãi vay: {self._format_number(data.get('interest_expense'))} VND
- Lợi nhuận trước thuế: {self._format_number(data.get('profit_before_tax'))} VND
- Thuế thu nhập doanh nghiệp: {self._format_number(data.get('tax_expense'))} VND
- Lợi nhuận sau thuế: {self._format_number(data.get('net_profit'))} VND
- Lợi nhuận của cổ đông: {self._format_number(data.get('net_profit_to_shareholders'))} VND
""".strip()

    def _create_balance_sheet_text(self, data: Dict) -> str:
        """Generate balance sheet text"""
        symbol = data['symbol']
        date = data.get('report_date', 'N/A')

        return f"""
Bảng cân đối kế toán của {symbol} tại ngày {date}:

TÀI SẢN:
- Tổng tài sản: {self._format_number(data.get('total_assets'))} VND
- Tài sản ngắn hạn: {self._format_number(data.get('current_assets'))} VND
- Tài sản dài hạn: {self._format_number(data.get('fixed_assets'))} VND

NỢ PHẢI TRẢ:
- Tổng nợ phải trả: {self._format_number(data.get('total_liabilities'))} VND
- Nợ ngắn hạn: {self._format_number(data.get('current_liabilities'))} VND
- Nợ dài hạn: {self._format_number(data.get('long_term_liabilities'))} VND

VỐN CHỦ SỞ HỮU:
- Vốn chủ sở hữu: {self._format_number(data.get('shareholders_equity'))} VND

Tỷ lệ thanh toán:
- Tỷ lệ thanh toán hiện hành: {data.get('current_ratio', 'N/A')}
- Tỷ lệ thanh toán nhanh: {data.get('quick_ratio', 'N/A')}
""".strip()

    def _create_cash_flow_text(self, data: Dict) -> str:
        """Generate cash flow statement"""
        symbol = data['symbol']
        period = (
            f"quý {data['fiscal_quarter']} năm {data['fiscal_year']}"
            if data.get('fiscal_quarter')
            else f"năm {data['fiscal_year']}"
        )

        return f"""
Báo cáo lưu chuyển tiền tệ của {symbol} trong {period}:

Hoạt động kinh doanh:
- Dòng tiền từ hoạt động kinh doanh: {self._format_number(data.get('operating_cash_flow'))} VND

Hoạt động đầu tư:
- Dòng tiền từ hoạt động đầu tư: {self._format_number(data.get('investing_cash_flow'))} VND

Hoạt động tài chính:
- Dòng tiền từ hoạt động tài chính: {self._format_number(data.get('financing_cash_flow'))} VND

Tổng hợp:
- Dòng tiền thuần trong kỳ: {self._format_number(data.get('net_cash_flow'))} VND
- Dòng tiền tự do (FCF): {self._format_number(data.get('free_cash_flow'))} VND
""".strip()

    def _create_ratio_analysis(self, data: Dict) -> str:
        """Create financial ratio analysis"""
        symbol = data['symbol']
        period = (
            f"quý {data['fiscal_quarter']} năm {data['fiscal_year']}"
            if data.get('fiscal_quarter')
            else f"năm {data['fiscal_year']}"
        )

        return f"""
Phân tích chỉ số tài chính của {symbol} trong {period}:

CHỈ SỐ SINH LỜI:
- EPS (Thu nhập trên mỗi cổ phiếu): {data.get('eps', 'N/A')} VND
- ROE (Lợi nhuận trên vốn chủ sở hữu): {data.get('roe', 'N/A')}%
- ROA (Lợi nhuận trên tổng tài sản): {data.get('roa', 'N/A')}%

CHỈ SỐ ĐÒN BẨY TÀI CHÍNH:
- Tỷ lệ nợ/Vốn chủ sở hữu: {data.get('debt_to_equity', 'N/A')}

CHỈ SỐ THANH KHOẢN:
- Hệ số thanh toán hiện hành: {data.get('current_ratio', 'N/A')}
- Hệ số thanh toán nhanh: {data.get('quick_ratio', 'N/A')}

Đánh giá:
{self._generate_assessment(data)}
""".strip()

    def _generate_assessment(self, data: Dict) -> str:
        """Generate simple assessment based on ratios"""
        assessments = []

        # ROE assessment
        roe = data.get('roe')
        if roe is not None:
            if roe > 15:
                assessments.append("ROE cao cho thấy hiệu quả sử dụng vốn tốt")
            elif roe > 10:
                assessments.append("ROE ở mức trung bình")
            else:
                assessments.append("ROE thấp cần cải thiện hiệu quả")

        # Debt ratio assessment
        d_to_e = data.get('debt_to_equity')
        if d_to_e is not None:
            if d_to_e < 1:
                assessments.append("Tỷ lệ nợ thấp, rủi ro tài chính thấp")
            elif d_to_e < 2:
                assessments.append("Tỷ lệ nợ ở mức chấp nhận được")
            else:
                assessments.append("Tỷ lệ nợ cao, cần theo dõi rủi ro")

        # Current ratio assessment
        current_ratio = data.get('current_ratio')
        if current_ratio is not None:
            if current_ratio > 2:
                assessments.append("Khả năng thanh toán ngắn hạn tốt")
            elif current_ratio > 1:
                assessments.append("Khả năng thanh toán ngắn hạn chấp nhận được")
            else:
                assessments.append("Khả năng thanh toán ngắn hạn yếu")

        return " - " + "\n - ".join(assessments) if assessments else "N/A"

    @staticmethod
    def _format_number(value) -> str:
        """Format large numbers in Vietnamese"""
        if value is None:
            return "N/A"
        try:
            num = float(value)
            if num >= 1_000_000_000_000:  # Trillion
                return f"{num/1_000_000_000_000:.2f} nghìn tỷ"
            elif num >= 1_000_000_000:  # Billion
                return f"{num/1_000_000_000:.2f} tỷ"
            elif num >= 1_000_000:  # Million
                return f"{num/1_000_000:.2f} triệu"
            elif num >= 1_000:
                return f"{num/1_000:.2f} nghìn"
            else:
                return f"{num:,.0f}"
        except (ValueError, TypeError):
            return "N/A"
