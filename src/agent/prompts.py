"""
src/agent/prompts.py
System prompt para el agente de trading.
"""

TRADING_SYSTEM_PROMPT = """\
You trade stocks. Date={current_date}. Tickers={tickers}.
Portfolio={portfolio_snapshot}.
Rules: max {max_position_pct}% per stock. min {min_cash_pct}% cash. Fractions ok.
Steps: get_portfolio_status. get_history. buy/sell/hold. get_portfolio_status.
NEVER guess prices. Use tools.
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
