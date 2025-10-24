"""
Field Mappings for Different Data Sources
=========================================

This module provides field name mappings between different data sources
and our standardized internal schema. Each data source may use different
field names (e.g., Vietnamese vs English, different naming conventions),
so we maintain mappings to normalize the data.

"""

from typing import Any, Dict, Optional, Union

import pandas as pd


class FieldMapper:
    """Base class for field mapping"""

    def __init__(self, mappings: Dict[str, str]):
        """
        Initialize field mapper

        Args:
            mappings: Dict mapping internal_field_name -> source_field_name
        """
        self.mappings = mappings
        self.reverse_mappings = {v: k for k, v in mappings.items()}

    def get_source_field(self, internal_field: str) -> Optional[str]:
        """Get source field name from internal field name"""
        return self.mappings.get(internal_field)

    def get_internal_field(self, source_field: str) -> Optional[str]:
        """Get internal field name from source field name"""
        return self.reverse_mappings.get(source_field)

    def extract_value(
        self,
        row: Union[pd.Series, Dict],
        internal_field: str,
        default: Any = None
    ) -> Any:
        """
        Extract value from row using internal field name

        Args:
            row: Pandas Series or dict containing the data
            internal_field: Our internal field name
            default: Default value if field not found

        Returns:
            Field value or default
        """
        source_field = self.get_source_field(internal_field)
        if source_field is None:
            return default

        if isinstance(row, pd.Series):
            return row.get(source_field, default)
        elif isinstance(row, dict):
            return row.get(source_field, default)
        else:
            return default

    def extract_numeric(
        self,
        row: Union[pd.Series, Dict],
        internal_field: str,
        default: Optional[float] = None
    ) -> Optional[float]:
        """
        Extract numeric value, handling None/NaN properly

        Args:
            row: Pandas Series or dict containing the data
            internal_field: Our internal field name
            default: Default value if field not found or invalid

        Returns:
            Float value or default
        """
        value = self.extract_value(row, internal_field, default)

        if value is None:
            return default

        # Handle pandas NaN
        if isinstance(value, float) and pd.isna(value):
            return default

        try:
            return float(value)
        except (ValueError, TypeError):
            return default


class VNStockIncomeStatementMapper(FieldMapper):
    """Field mapper for VNStock Income Statement (Vietnamese)"""

    def __init__(self):
        mappings = {
            # Basic identifiers
            "symbol": "CP",
            "fiscal_year": "Năm",
            "fiscal_quarter": "Kỳ",

            # Revenue & Growth
            "revenue": "Doanh thu (đồng)",
            "revenue_growth": "Tăng trưởng doanh thu (%)",
            "net_revenue": "Doanh thu thuần",

            # Profits
            "gross_profit": "Lãi gộp",
            "operating_profit": "Lãi/Lỗ từ hoạt động kinh doanh",
            "profit_before_tax": "LN trước thuế",
            "net_profit": "Lợi nhuận thuần",
            "net_profit_to_shareholders": (
                "Lợi nhuận sau thuế của Cổ đông công ty mẹ (đồng)"
            ),
            "profit_growth": "Tăng trưởng lợi nhuận (%)",

            # Costs & Expenses
            "cost_of_goods_sold": "Giá vốn hàng bán",
            "selling_expenses": "Chi phí bán hàng",
            "admin_expenses": "Chi phí quản lý DN",
            "operating_expenses": "Chi phí bán hàng",  # Approximation
            "financial_expenses": "Chi phí tài chính",
            "interest_expense": "Chi phí tiền lãi vay",

            # Income
            "financial_income": "Thu nhập tài chính",
            "other_income": "Thu nhập khác",

            # Tax
            "current_tax_expense": "Chi phí thuế TNDN hiện hành",
            "deferred_tax_expense": "Chi phí thuế TNDN hoãn lại",

            # Special items
            "minority_interest": "Cổ đông thiểu số",
        }
        super().__init__(mappings)


class VNStockBalanceSheetMapper(FieldMapper):
    """Field mapper for VNStock Balance Sheet (Vietnamese)"""

    def __init__(self):
        mappings = {
            # Basic identifiers
            "symbol": "CP",
            "fiscal_year": "Năm",
            "fiscal_quarter": "Kỳ",

            # Assets
            "total_assets": "TỔNG CỘNG TÀI SẢN (đồng)",
            "current_assets": "TÀI SẢN NGẮN HẠN (đồng)",
            "fixed_assets": "TÀI SẢN DÀI HẠN (đồng)",
            "cash_and_equivalents": "Tiền và tương đương tiền (đồng)",
            "short_term_investments": "Giá trị thuần đầu tư ngắn hạn (đồng)",
            "accounts_receivable": "Các khoản phải thu ngắn hạn (đồng)",
            "inventory": "Hàng tồn kho ròng",
            "other_current_assets": "Tài sản lưu động khác",
            "ppe_net": "Tài sản cố định (đồng)",
            "long_term_investments": "Đầu tư dài hạn (đồng)",
            "goodwill": "Lợi thế thương mại",

            # Liabilities
            "total_liabilities": "NỢ PHẢI TRẢ (đồng)",
            "current_liabilities": "Nợ ngắn hạn (đồng)",
            "long_term_liabilities": "Nợ dài hạn (đồng)",
            "short_term_borrowings": "Vay và nợ thuê tài chính ngắn hạn (đồng)",
            "long_term_borrowings": "Vay và nợ thuê tài chính dài hạn (đồng)",

            # Equity
            "shareholders_equity": "VỐN CHỦ SỞ HỮU (đồng)",
            "share_capital": "Vốn góp của chủ sở hữu (đồng)",
            "retained_earnings": "Lãi chưa phân phối (đồng)",
            "minority_interest": "LỢI ÍCH CỦA CỔ ĐÔNG THIỂU SỐ",

            # Total
            "total_equity_and_liabilities": "TỔNG CỘNG NGUỒN VỐN (đồng)",
        }
        super().__init__(mappings)


class VNStockCashFlowMapper(FieldMapper):
    """Field mapper for VNStock Cash Flow Statement (Vietnamese)"""

    def __init__(self):
        mappings = {
            # Basic identifiers
            "symbol": "CP",
            "fiscal_year": "Năm",
            "fiscal_quarter": "Kỳ",

            # Operating Activities
            "operating_cash_flow": "Lưu chuyển tiền thuần từ HĐ kinh doanh",
            "cash_from_operations": "Tiền thu từ bán hàng, cung cấp dịch vụ",

            # Investing Activities
            "investing_cash_flow": "Lưu chuyển tiền thuần từ HĐ đầu tư",
            "capex": "Tiền chi mua sắm TSCĐ",
            "cash_from_investments": "Tiền thu từ thanh lý TSCĐ",

            # Financing Activities
            "financing_cash_flow": "Lưu chuyển tiền thuần từ HĐ tài chính",
            "cash_from_borrowings": "Tiền thu từ vay",
            "cash_to_dividends": "Tiền chi trả cổ tức",

            # Net Change
            "net_cash_flow": "Lưu chuyển tiền thuần trong kỳ",
            "cash_beginning": "Tiền và tương đương tiền đầu kỳ",
            "cash_ending": "Tiền và tương đương tiền cuối kỳ",
        }
        super().__init__(mappings)


class VNStockRatioMapper(FieldMapper):
    """Field mapper for VNStock Financial Ratios (Vietnamese)"""

    def __init__(self):
        # Note: VNStock ratios use multi-level columns (tuples)
        mappings = {
            # Basic identifiers
            "symbol": ("Meta", "CP"),
            "fiscal_year": ("Meta", "Năm"),
            "fiscal_quarter": ("Meta", "Kỳ"),

            # Profitability Ratios
            "roe": ("Chỉ tiêu khả năng sinh lợi", "ROE (%)"),
            "roa": ("Chỉ tiêu khả năng sinh lợi", "ROA (%)"),
            "roic": ("Chỉ tiêu khả năng sinh lợi", "ROIC (%)"),
            "gross_margin": (
                "Chỉ tiêu khả năng sinh lợi",
                "Biên lợi nhuận gộp (%)"
            ),
            "operating_margin": (
                "Chỉ tiêu khả năng sinh lợi",
                "Biên EBIT (%)"
            ),
            "net_margin": (
                "Chỉ tiêu khả năng sinh lợi",
                "Biên lợi nhuận ròng (%)"
            ),
            "ebitda": ("Chỉ tiêu khả năng sinh lợi", "EBITDA (Tỷ đồng)"),
            "ebit": ("Chỉ tiêu khả năng sinh lợi", "EBIT (Tỷ đồng)"),

            # Liquidity Ratios
            "current_ratio": (
                "Chỉ tiêu thanh khoản",
                "Chỉ số thanh toán hiện thời"
            ),
            "quick_ratio": (
                "Chỉ tiêu thanh khoản",
                "Chỉ số thanh toán nhanh"
            ),
            "cash_ratio": (
                "Chỉ tiêu thanh khoản",
                "Chỉ số thanh toán tiền mặt"
            ),

            # Leverage Ratios
            "debt_to_equity": (
                "Chỉ tiêu cơ cấu nguồn vốn",
                "Nợ/VCSH"
            ),
            "debt_ratio": (
                "Chỉ tiêu cơ cấu nguồn vốn",
                "(Vay NH+DH)/VCSH"
            ),
            "financial_leverage": (
                "Chỉ tiêu thanh khoản",
                "Đòn bẩy tài chính"
            ),

            # Efficiency Ratios
            "asset_turnover": (
                "Chỉ tiêu hiệu quả hoạt động",
                "Vòng quay tài sản"
            ),
            "inventory_turnover": (
                "Chỉ tiêu hiệu quả hoạt động",
                "Vòng quay hàng tồn kho"
            ),
            "receivable_days": (
                "Chỉ tiêu hiệu quả hoạt động",
                "Số ngày thu tiền bình quân"
            ),
            "inventory_days": (
                "Chỉ tiêu hiệu quả hoạt động",
                "Số ngày tồn kho bình quân"
            ),
            "payable_days": (
                "Chỉ tiêu hiệu quả hoạt động",
                "Số ngày thanh toán bình quân"
            ),

            # Valuation Ratios
            "pe_ratio": ("Chỉ tiêu định giá", "P/E"),
            "pb_ratio": ("Chỉ tiêu định giá", "P/B"),
            "ps_ratio": ("Chỉ tiêu định giá", "P/S"),
            "eps": ("Chỉ tiêu định giá", "EPS (VND)"),
            "bvps": ("Chỉ tiêu định giá", "BVPS (VND)"),
            "market_cap": ("Chỉ tiêu định giá", "Vốn hóa (Tỷ đồng)"),
        }
        super().__init__(mappings)


class SourceFieldMapperFactory:
    """Factory to get appropriate field mapper for data source"""

    _mappers = {
        "vnstock_income": VNStockIncomeStatementMapper,
        "vnstock_balance": VNStockBalanceSheetMapper,
        "vnstock_cashflow": VNStockCashFlowMapper,
        "vnstock_ratio": VNStockRatioMapper,
    }

    @classmethod
    def get_mapper(cls, source_type: str) -> FieldMapper:
        """
        Get field mapper for given source type

        Args:
            source_type: Type of source data
                (e.g., 'vnstock_income', 'vnstock_balance')

        Returns:
            FieldMapper instance

        Raises:
            ValueError: If source_type not recognized
        """
        mapper_class = cls._mappers.get(source_type)
        if mapper_class is None:
            raise ValueError(
                f"Unknown source type: {source_type}. "
                f"Available: {list(cls._mappers.keys())}"
            )
        return mapper_class()

    @classmethod
    def register_mapper(cls, source_type: str, mapper_class: type):
        """
        Register a new field mapper

        Args:
            source_type: Identifier for the source type
            mapper_class: FieldMapper subclass
        """
        cls._mappers[source_type] = mapper_class

    @classmethod
    def list_available_mappers(cls) -> list:
        """Get list of available mapper types"""
        return list(cls._mappers.keys())
