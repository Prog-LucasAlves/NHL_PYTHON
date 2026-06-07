def prediction_card_html(
    home_abbr: str,
    away_abbr: str,
    prob_home: float,
    prob_away: float,
    fair_odd_home: float,
    fair_odd_away: float,
) -> str:
    """Gera o HTML do card de predição VS."""
    return f"""
    <div class="prediction-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="text-align: center; flex: 1;">
                <h2 style="color: #ffffff; margin-bottom: 0;">{home_abbr}</h2>
                <h1 style="color: #00ff88; font-size: 4rem; margin-top: 0;">{prob_home:.1%}</h1>
                <p style="color: #888;">Odd Justa: <b>{fair_odd_home:.2f}</b></p>
            </div>
            <div style="font-size: 3rem; color: #444; padding: 0 20px;">VS</div>
            <div style="text-align: center; flex: 1;">
                <h2 style="color: #ffffff; margin-bottom: 0;">{away_abbr}</h2>
                <h1 style="color: #00bdff; font-size: 4rem; margin-top: 0;">{prob_away:.1%}</h1>
                <p style="color: #888;">Odd Justa: <b>{fair_odd_away:.2f}</b></p>
            </div>
        </div>
    </div>
    """


def bet_register_header_html() -> str:
    """Header estilizado da seção de registro de aposta."""
    return """
    <div class="bet-register-header">
        <h3 style="color: #00bdff; margin-top: 0;">📝 Registrar Aposta</h3>
    </div>
    """
