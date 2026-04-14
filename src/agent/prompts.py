"""
src/agent/prompts.py
System prompt for the trading agent.
"""

TRADING_SYSTEM_PROMPT = """\
DATE: {current_date}
TICKERS: {tickers}
PORTFOLIO: {portfolio_snapshot}
RULES: max {max_position_pct}% one ticker. keep {min_cash_pct}% cash. fractions ok.

DO THIS. IN ORDER. NO SKIP:
1. CALL get_portfolio_status
2. CALL get_history FOR EACH TICKER. ONE CALL PER TICKER. ALL TICKERS.
3. LOOK DATA. DECIDE: BUY or SELL or HOLD each ticker.
4. CALL execute_buy or execute_sell. CAN DO MANY TRADES.
5. CALL get_portfolio_status again.
6. WRITE SHORT SUMMARY.

NO GUESS PRICE. USE TOOL.
NO TALK BETWEEN TOOL CALLS.
DIVERSIFY. NOT ONLY ONE TICKER.
"""


def build_system_prompt(
    current_date: str,
    portfolio_snapshot: dict,
    tickers: list[str],
    max_position_pct: float = 0.25,
    min_cash_pct: float = 0.05,
) -> str:
    return TRADING_SYSTEM_PROMPT.format(
        current_date=current_date,
        portfolio_snapshot=str(portfolio_snapshot),
        tickers=", ".join(tickers),
        max_position_pct=int(max_position_pct * 100),
        min_cash_pct=int(min_cash_pct * 100),
    )
