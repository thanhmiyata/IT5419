"""
Financial Strength Indicators
==============================

Financial strength and solvency indicators for evaluating company health:
- Liquidity Ratios
- Solvency Ratios
- Efficiency Ratios
- Cash Flow Indicators
- Financial Health Scores
"""

from typing import Dict, Optional


class FinancialStrengthIndicators:
    """Calculate financial strength and solvency indicators."""

    # Liquidity Ratios
    @staticmethod
    def current_ratio(current_assets: float, current_liabilities: float) -> Optional[float]:
        """
        Calculate Current Ratio (ability to pay short-term obligations).

        Args:
            current_assets: Total current assets
            current_liabilities: Total current liabilities

        Returns:
            Current ratio (> 1.0 is healthy)
        """
        if current_liabilities == 0:
            return None
        return current_assets / current_liabilities

    @staticmethod
    def quick_ratio(current_assets: float, inventory: float, current_liabilities: float) -> Optional[float]:
        """
        Calculate Quick Ratio (Acid-Test Ratio).

        Args:
            current_assets: Total current assets
            inventory: Inventory value
            current_liabilities: Total current liabilities

        Returns:
            Quick ratio (> 1.0 is healthy)
        """
        if current_liabilities == 0:
            return None
        return (current_assets - inventory) / current_liabilities

    @staticmethod
    def cash_ratio(cash: float, cash_equivalents: float, current_liabilities: float) -> Optional[float]:
        """
        Calculate Cash Ratio (most conservative liquidity measure).

        Args:
            cash: Cash on hand
            cash_equivalents: Marketable securities
            current_liabilities: Total current liabilities

        Returns:
            Cash ratio
        """
        if current_liabilities == 0:
            return None
        return (cash + cash_equivalents) / current_liabilities

    @staticmethod
    def working_capital(current_assets: float, current_liabilities: float) -> float:
        """
        Calculate Working Capital.

        Args:
            current_assets: Total current assets
            current_liabilities: Total current liabilities

        Returns:
            Working capital amount
        """
        return current_assets - current_liabilities

    # Solvency Ratios
    @staticmethod
    def debt_service_coverage_ratio(operating_income: float, total_debt_service: float) -> Optional[float]:
        """
        Calculate Debt Service Coverage Ratio (DSCR).

        Args:
            operating_income: Net operating income
            total_debt_service: Total debt obligations (principal + interest)

        Returns:
            DSCR (> 1.0 means can cover debt)
        """
        if total_debt_service == 0:
            return None
        return operating_income / total_debt_service

    @staticmethod
    def equity_multiplier(total_assets: float, shareholders_equity: float) -> Optional[float]:
        """
        Calculate Equity Multiplier (financial leverage).

        Args:
            total_assets: Total assets
            shareholders_equity: Total shareholder equity

        Returns:
            Equity multiplier
        """
        if shareholders_equity == 0:
            return None
        return total_assets / shareholders_equity

    @staticmethod
    def times_interest_earned(ebit: float, interest_expense: float) -> Optional[float]:
        """
        Calculate Times Interest Earned (Interest Coverage).

        Args:
            ebit: Earnings before interest and taxes
            interest_expense: Interest expense

        Returns:
            TIE ratio (higher is better)
        """
        if interest_expense == 0:
            return None
        return ebit / interest_expense

    # Efficiency Ratios
    @staticmethod
    def asset_turnover(revenue: float, total_assets: float) -> Optional[float]:
        """
        Calculate Asset Turnover Ratio.

        Args:
            revenue: Total revenue
            total_assets: Total assets

        Returns:
            Asset turnover (higher is better)
        """
        if total_assets == 0:
            return None
        return revenue / total_assets

    @staticmethod
    def inventory_turnover(cogs: float, average_inventory: float) -> Optional[float]:
        """
        Calculate Inventory Turnover Ratio.

        Args:
            cogs: Cost of goods sold
            average_inventory: Average inventory

        Returns:
            Inventory turnover
        """
        if average_inventory == 0:
            return None
        return cogs / average_inventory

    @staticmethod
    def receivables_turnover(revenue: float, average_receivables: float) -> Optional[float]:
        """
        Calculate Receivables Turnover Ratio.

        Args:
            revenue: Total revenue
            average_receivables: Average accounts receivable

        Returns:
            Receivables turnover
        """
        if average_receivables == 0:
            return None
        return revenue / average_receivables

    @staticmethod
    def days_sales_outstanding(receivables_turnover: float) -> Optional[float]:
        """
        Calculate Days Sales Outstanding (DSO).

        Args:
            receivables_turnover: Receivables turnover ratio

        Returns:
            DSO in days (lower is better)
        """
        if receivables_turnover == 0:
            return None
        return 365 / receivables_turnover

    @staticmethod
    def days_inventory_outstanding(inventory_turnover: float) -> Optional[float]:
        """
        Calculate Days Inventory Outstanding (DIO).

        Args:
            inventory_turnover: Inventory turnover ratio

        Returns:
            DIO in days
        """
        if inventory_turnover == 0:
            return None
        return 365 / inventory_turnover

    # Cash Flow Indicators
    @staticmethod
    def operating_cash_flow_ratio(operating_cash_flow: float, current_liabilities: float) -> Optional[float]:
        """
        Calculate Operating Cash Flow Ratio.

        Args:
            operating_cash_flow: Cash flow from operations
            current_liabilities: Total current liabilities

        Returns:
            OCF ratio (> 1.0 is healthy)
        """
        if current_liabilities == 0:
            return None
        return operating_cash_flow / current_liabilities

    @staticmethod
    def cash_flow_to_debt(operating_cash_flow: float, total_debt: float) -> Optional[float]:
        """
        Calculate Cash Flow to Debt Ratio.

        Args:
            operating_cash_flow: Cash flow from operations
            total_debt: Total debt

        Returns:
            CF/Debt ratio (higher is better)
        """
        if total_debt == 0:
            return None
        return operating_cash_flow / total_debt

    @staticmethod
    def free_cash_flow(operating_cash_flow: float, capital_expenditures: float) -> float:
        """
        Calculate Free Cash Flow (FCF).

        Args:
            operating_cash_flow: Cash flow from operations
            capital_expenditures: Capital expenditures

        Returns:
            FCF amount
        """
        return operating_cash_flow - capital_expenditures

    @staticmethod
    def cash_conversion_cycle(dso: float, dio: float, dpo: float) -> float:
        """
        Calculate Cash Conversion Cycle (CCC).

        Args:
            dso: Days Sales Outstanding
            dio: Days Inventory Outstanding
            dpo: Days Payable Outstanding

        Returns:
            CCC in days (lower is better)
        """
        return dso + dio - dpo

    # Financial Health Scores
    @staticmethod
    def altman_z_score(
        working_capital: float,
        retained_earnings: float,
        ebit: float,
        market_value_equity: float,
        total_assets: float,
        total_liabilities: float,
        sales: float
    ) -> float:
        """
        Calculate Altman Z-Score (bankruptcy prediction).

        Args:
            working_capital: Working capital
            retained_earnings: Retained earnings
            ebit: Earnings before interest and taxes
            market_value_equity: Market value of equity
            total_assets: Total assets
            total_liabilities: Total liabilities
            sales: Total sales

        Returns:
            Z-Score (> 2.99 = safe, 1.81-2.99 = grey, < 1.81 = distress)
        """
        if total_assets == 0:
            return 0.0

        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_value_equity / total_liabilities if total_liabilities > 0 else 0
        x5 = sales / total_assets

        z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        return z_score

    @staticmethod
    def piotroski_f_score(
        net_income: float,
        operating_cash_flow: float,
        roa_current: float,
        roa_previous: float,
        total_assets_current: float,
        total_assets_previous: float,
        long_term_debt_current: float,
        long_term_debt_previous: float,
        current_ratio_current: float,
        current_ratio_previous: float,
        shares_outstanding_current: float,
        shares_outstanding_previous: float,
        gross_margin_current: float,
        gross_margin_previous: float,
        asset_turnover_current: float,
        asset_turnover_previous: float
    ) -> int:
        """
        Calculate Piotroski F-Score (financial strength score 0-9).

        Returns:
            F-Score (8-9 = strong, 0-2 = weak)
        """
        score = 0

        # Profitability signals
        if net_income > 0:
            score += 1
        if operating_cash_flow > 0:
            score += 1
        if roa_current > roa_previous:
            score += 1
        if operating_cash_flow > net_income:
            score += 1

        # Leverage, Liquidity, and Source of Funds
        if long_term_debt_current < long_term_debt_previous:
            score += 1
        if current_ratio_current > current_ratio_previous:
            score += 1
        if shares_outstanding_current <= shares_outstanding_previous:
            score += 1

        # Operating Efficiency
        if gross_margin_current > gross_margin_previous:
            score += 1
        if asset_turnover_current > asset_turnover_previous:
            score += 1

        return score

    @staticmethod
    def beneish_m_score(
        receivables: float,
        revenue: float,
        cogs: float,
        sga: float,
        depreciation: float,
        total_assets: float,
        current_assets: float,
        current_liabilities: float,
        long_term_debt: float,
        receivables_prev: float,
        revenue_prev: float,
        cogs_prev: float,
        sga_prev: float,
        total_assets_prev: float,
        current_assets_prev: float,
        current_liabilities_prev: float
    ) -> Optional[float]:
        """
        Calculate Beneish M-Score (earnings manipulation detector).

        Returns:
            M-Score (> -2.22 suggests manipulation)
        """
        if revenue_prev == 0 or total_assets_prev == 0:
            return None

        # Days Sales in Receivables Index
        dsri = (receivables / revenue) / (receivables_prev / revenue_prev) if revenue_prev > 0 else 1

        # Gross Margin Index
        gmi = ((revenue_prev - cogs_prev) / revenue_prev) / ((revenue - cogs) / revenue) if revenue > 0 else 1

        # Asset Quality Index
        aqi = (1 - (current_assets + depreciation) / total_assets) / \
              (1 - (current_assets_prev + depreciation) / total_assets_prev)

        # Sales Growth Index
        sgi = revenue / revenue_prev if revenue_prev > 0 else 1

        # Depreciation Index
        depi = (depreciation / (depreciation + total_assets_prev)) / \
               (depreciation / (depreciation + total_assets))

        # SG&A Index
        sgai = (sga / revenue) / (sga_prev / revenue_prev) if revenue_prev > 0 and revenue > 0 else 1

        # Leverage Index
        lvgi = ((current_liabilities + long_term_debt) / total_assets) / \
               ((current_liabilities_prev + long_term_debt) / total_assets_prev)

        # Total Accruals to Total Assets
        tata = ((revenue - cogs - sga - depreciation
                 - (current_assets - receivables - current_liabilities)) / total_assets)

        m_score = (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi
                   + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi)

        return m_score

    @classmethod
    def calculate_all_strength_indicators(
        cls,
        financial_data: Dict[str, float]
    ) -> Dict[str, Optional[float]]:
        """Calculate all financial strength indicators at once."""
        results: Dict[str, Optional[float]] = {}

        # Liquidity Ratios
        results["current_ratio"] = cls.current_ratio(
            financial_data.get("current_assets", 0),
            financial_data.get("current_liabilities", 0)
        )
        results["quick_ratio"] = cls.quick_ratio(
            financial_data.get("current_assets", 0),
            financial_data.get("inventory", 0),
            financial_data.get("current_liabilities", 0)
        )
        results["cash_ratio"] = cls.cash_ratio(
            financial_data.get("cash", 0),
            financial_data.get("cash_equivalents", 0),
            financial_data.get("current_liabilities", 0)
        )
        results["working_capital"] = cls.working_capital(
            financial_data.get("current_assets", 0),
            financial_data.get("current_liabilities", 0)
        )

        # Efficiency Ratios
        results["asset_turnover"] = cls.asset_turnover(
            financial_data.get("revenue", 0),
            financial_data.get("total_assets", 0)
        )

        # Cash Flow
        results["operating_cash_flow_ratio"] = cls.operating_cash_flow_ratio(
            financial_data.get("operating_cash_flow", 0),
            financial_data.get("current_liabilities", 0)
        )
        results["free_cash_flow"] = cls.free_cash_flow(
            financial_data.get("operating_cash_flow", 0),
            financial_data.get("capex", 0)
        )

        return results
