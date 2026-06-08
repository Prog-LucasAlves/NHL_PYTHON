import os
import sys
import zoneinfo
from datetime import datetime

import requests

from nhl_engine.config import TEAM_MAPPING

# Mapeamentos para data formatada em português brasileiro sem dependência de locales do sistema
DIAS_SEMANA = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo",
}

MESES = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}


def get_nhl_schedule(date_str: str) -> list[dict]:
    """Busca a lista de jogos da NHL para uma data específica (YYYY-MM-DD)."""
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            print(f"Erro na requisição para a API da NHL: Status {response.status_code}")
            return []

        data = response.json()
        game_week = data.get("gameWeek", [])

        # Procura o dia específico correspondente à data solicitada
        for day in game_week:
            if day.get("date") == date_str:
                return day.get("games", [])

        return []
    except Exception as e:
        print(f"Exceção ao buscar cronograma da NHL: {e}")
        return []


def format_message(games: list[dict], date_str: str) -> str:
    """Formata a lista de jogos em uma mensagem amigável no formato HTML para o Telegram."""
    # Gera a data formatada: ex: Domingo, 7 de Junho de 2026
    try:
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
        dia_semana = DIAS_SEMANA[dt_obj.weekday()]
        dia = dt_obj.day
        mes = MESES[dt_obj.month]
        ano = dt_obj.year
        formatted_date = f"{dia_semana}, {dia} de {mes} de {ano}"
    except Exception:
        formatted_date = date_str

    header = f"🏒 <b>NHL - JOGOS DE HOJE</b> 🏒\n📅 <i>{formatted_date}</i>\n_______________________\n"

    footer = "_______________________\n\n<i>Notificação automática gerada pelo NHL Stats Predictor.</i>"

    if not games:
        body = "\n😴 Não há partidas programadas para a rodada de hoje.\n"
        return header + body + footer

    body_lines = [""]
    for game in games:
        away_abbr = game.get("awayTeam", {}).get("abbrev", "Away")
        home_abbr = game.get("homeTeam", {}).get("abbrev", "Home")

        away_name = TEAM_MAPPING.get(away_abbr, away_abbr)
        home_name = TEAM_MAPPING.get(home_abbr, home_abbr)

        # Converte horário de UTC para Brasília
        start_time_utc = game.get("startTimeUTC")
        game_time_str = "Horário não disponível"
        if start_time_utc:
            try:
                # O formato do startTimeUTC é ISO 8601 (ex: 2026-06-10T00:00:00Z)
                dt_utc = datetime.fromisoformat(start_time_utc.replace("Z", "+00:00"))
                br_tz = zoneinfo.ZoneInfo("America/Sao_Paulo")
                dt_br = dt_utc.astimezone(br_tz)
                game_time_str = dt_br.strftime("%H:%M") + " BRT"
            except Exception as e:
                print(f"Erro ao converter horário {start_time_utc}: {e}")

        # Formata o link do Game Center
        game_center_raw = game.get("gameCenterLink")
        game_link_html = ""
        if game_center_raw:
            game_center_url = f"https://www.nhl.com{game_center_raw}"
            game_link_html = f' | <a href="{game_center_url}">Game Center</a>'

        # Verifica se há informações de playoffs / série
        series_info = ""
        series_status = game.get("seriesStatus")
        if series_status:
            series_title = series_status.get("seriesTitle", "")
            game_number = series_status.get("gameNumberOfSeries", "")
            if series_title and game_number:
                series_info = f"\n🏆 <i>{series_title} - Jogo {game_number}</i>"

        body_lines.append(f"✈️ <b>{away_name}</b> vs 🏠 <b>{home_name}</b>")
        body_lines.append(f"⏰ {game_time_str}{game_link_html}{series_info}")
        body_lines.append("")

    body = "\n".join(body_lines)
    return header + body + footer


def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    """Envia uma mensagem formatada em HTML para o grupo do Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML", "disable_web_page_preview": True}
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            print("Notificação enviada ao Telegram com sucesso!")
            return True
        else:
            print(f"Erro ao enviar notificação ao Telegram: Status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exceção ao enviar mensagem ao Telegram: {e}")
        return False


def main() -> None:
    # 1. Recupera as credenciais do Telegram das variáveis de ambiente
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Erro: As variáveis de ambiente TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID são obrigatórias.", file=sys.stderr)
        sys.exit(1)

    # 2. Obtém a data de hoje no horário de Brasília (BRT)
    br_tz = zoneinfo.ZoneInfo("America/Sao_Paulo")
    now_br = datetime.now(br_tz)
    today_str = now_br.strftime("%Y-%m-%d")

    print(f"Iniciando verificação de jogos para a data (Brasília): {today_str}")

    # 3. Busca os jogos do dia na API da NHL
    games = get_nhl_schedule(today_str)
    print(f"Quantidade de jogos encontrados: {len(games)}")

    # 4. Formata a mensagem
    message = format_message(games, today_str)

    # 5. Envia a notificação para o grupo do Telegram
    success = send_telegram_message(token, chat_id, message)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
