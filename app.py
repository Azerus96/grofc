from flask import Flask, render_template, jsonify, session, request
import os
import asyncio
import aiohttp
import base64
import time
import json
from threading import Thread, Event
import logging
from ai_engine import CFRAgent, RandomAgent, Card, Board, GameState

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

cfr_agent = None
random_agent = RandomAgent()
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "Azerus96")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "grofc")
AI_PROGRESS_FILENAME = "cfr_data.pkl"

def initialize_ai_agent(ai_settings):
    """Инициализирует AI агента с учётом сохранённого прогресса."""
    global cfr_agent
    iterations = int(ai_settings.get("iterations", 500000))
    stop_threshold = float(ai_settings.get("stopThreshold", 0.0001))
    if ai_settings.get("aiType") == "mccfr":
        cfr_agent = CFRAgent(iterations=iterations, stop_threshold=stop_threshold)
        if os.environ.get("AI_PROGRESS_TOKEN"):
            asyncio.run(load_progress_async())
            data = utils.load_ai_progress("cfr_data.pkl")
            if data:
                cfr_agent.nodes = data["nodes"]
                cfr_agent.iterations = data["iterations"]
                cfr_agent.stop_threshold = data.get("stop_threshold", 0.0001)
                logger.info("Прогресс AI загружен.")
        logger.info(f"MCCFR агент инициализирован с iterations={iterations}")

async def save_progress_async():
    """Асинхронно сохраняет прогресс на GitHub."""
    if cfr_agent:
        cfr_agent.save_progress()
        async with aiohttp.ClientSession() as session:
            token = os.environ.get("AI_PROGRESS_TOKEN")
            if not token:
                logger.warning("Токен GitHub отсутствует.")
                return
            url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPOSITORY}/contents/{AI_PROGRESS_FILENAME}"
            with open("cfr_data.pkl", "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")
            headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
            try:
                async with session.get(url, headers=headers) as resp:
                    sha = (await resp.json()).get("sha") if resp.status == 200 else None
                data = {
                    "message": f"Update AI progress ({time.strftime('%Y-%m-%d %H:%M:%S')})",
                    "content": content,
                    "branch": "main",
                    "sha": sha
                }
                async with session.put(url, json=data, headers=headers) as resp:
                    logger.info("Прогресс сохранён на GitHub." if resp.status in (200, 201) else f"Ошибка GitHub: {await resp.text()}")
            except Exception as e:
                logger.error(f"Ошибка сохранения: {e}")

async def load_progress_async():
    """Асинхронно загружает прогресс с GitHub."""
    async with aiohttp.ClientSession() as session:
        token = os.environ.get("AI_PROGRESS_TOKEN")
        if not token:
            return
        url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{GITHUB_REPOSITORY}/contents/{AI_PROGRESS_FILENAME}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                content = base64.b64decode((await resp.json())["content"])
                with open("cfr_data.pkl", "wb") as f:
                    f.write(content)
                logger.info("Прогресс загружен с GitHub.")

def serialize_card(card):
    return card.to_dict() if card else None

def serialize_move(move):
    return {key: [serialize_card(card) for card in cards] if isinstance(cards, list) else serialize_card(cards) for key, cards in move.items()}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/training")
def training():
    session["game_state"] = {
        "selected_cards": [],
        "board": {"top": [None] * 3, "middle": [None] * 5, "bottom": [None] * 5},
        "discarded_cards": [],
        "ai_settings": {
            "fantasyType": "normal",
            "fantasyMode": False,
            "aiTime": "60",
            "iterations": "500000",
            "stopThreshold": "0.0001",
            "aiType": "mccfr",
            "placementMode": "standard"
        }
    }
    if session.get("previous_ai_settings") != session["game_state"]["ai_settings"]:
        initialize_ai_agent(session["game_state"]["ai_settings"])
        session["previous_ai_settings"] = session["game_state"]["ai_settings"].copy()
    return render_template("training.html", game_state=session["game_state"])

@app.route("/update_state", methods=["POST"])
def update_state():
    if not request.is_json:
        return jsonify({"error": "Content type must be application/json"}), 400
    try:
        game_state = request.get_json()
        if "game_state" not in session:
            session["game_state"] = {}
        
        if "board" in game_state:
            current_board = session["game_state"].get("board", {"top": [None] * 3, "middle": [None] * 5, "bottom": [None] * 5})
            for line in ["top", "middle", "bottom"]:
                if line in game_state["board"]:
                    new_line = game_state["board"][line]
                    current_line = current_board[line]
                    for i, card in enumerate(new_line):
                        if i < len(current_line) and card:
                            current_line[i] = Card.from_dict(card)
                    current_board[line] = current_line
            session["game_state"]["board"] = current_board

        for key in ["selected_cards", "discarded_cards"]:
            if key in game_state:
                session["game_state"][key] = [Card.from_dict(card) for card in game_state[key] if card]

        if "removed_cards" in game_state:
            removed = [Card.from_dict(card) for card in game_state["removed_cards"]]
            session["game_state"]["discarded_cards"] = list(set(session["game_state"].get("discarded_cards", []) + removed))
            session["game_state"]["selected_cards"] = [c for c in session["game_state"].get("selected_cards", []) if c not in removed]

        if "ai_settings" in game_state:
            session["game_state"]["ai_settings"] = game_state["ai_settings"]
            if game_state["ai_settings"] != session.get("previous_ai_settings"):
                initialize_ai_agent(game_state["ai_settings"])
                session["previous_ai_settings"] = game_state["ai_settings"].copy()

        return jsonify({"status": "success"})
    except Exception as e:
        logger.exception(f"Ошибка в update_state: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ai_move", methods=["POST"])
async def ai_move():
    global cfr_agent, random_agent
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid game state data"}), 400

    selected_cards = [Card.from_dict(c) for c in data.get("selected_cards", [])]
    discarded_cards = [Card.from_dict(c) for c in data.get("discarded_cards", [])]
    board = Board()
    for line in ["top", "middle", "bottom"]:
        for card_data in data.get("board", {}).get(line, []):
            if card_data:
                board.place_card(line, Card.from_dict(card_data))

    game_state = GameState(selected_cards=selected_cards, board=board, discarded_cards=discarded_cards, ai_settings=data.get("ai_settings", {}))
    if game_state.is_terminal():
        royalties = game_state.calculate_royalties()
        total_royalty = sum(royalties.values()) if isinstance(royalties, dict) else 0
        if cfr_agent and data.get("ai_settings", {}).get("aiType") == "mccfr":
            await save_progress_async()
        return jsonify({"message": "Game over", "royalties": royalties, "total_royalty": total_royalty, "game_over": True})

    timeout_event = Event()
    result = {"move": None}
    ai_type = data.get("ai_settings", {}).get("aiType", "mccfr")
    ai_thread = Thread(target=(cfr_agent if ai_type == "mccfr" else random_agent).get_move, args=(game_state, timeout_event, result))
    ai_thread.start()
    ai_thread.join(timeout=int(data.get("ai_settings", {}).get("aiTime", 5)))

    if ai_thread.is_alive():
        timeout_event.set()
        ai_thread.join()
        return jsonify({"error": "AI move timed out"}), 504

    move = result.get("move")
    if move is None or "error" in move:
        return jsonify({"error": move.get("error", "Unknown error")}), 500

    next_game_state = game_state.apply_action(move)
    if cfr_agent and ai_type == "mccfr":
        await save_progress_async()

    if next_game_state.is_terminal():
        royalties = next_game_state.calculate_royalties()
        total_royalty = sum(royalties.values()) if isinstance(royalties, dict) else 0
        return jsonify({"move": serialize_move(move), "royalties": royalties, "total_royalty": total_royalty, "game_over": True})
    return jsonify({"move": serialize_move(move), "royalties": {}, "total_royalty": 0})

@app.route("/play", methods=["POST"])
async def play():
    data = request.get_json()
    player_board = Board()
    for line in ["top", "middle", "bottom"]:
        for card_data in data["board"][line]:
            if card_data:
                player_board.place_card(line, Card.from_dict(card_data))

    game_state = GameState(
        selected_cards=Hand([Card.from_dict(c) for c in data["selected_cards"]]),
        board=player_board,
        discarded_cards=[Card.from_dict(c) for c in data.get("discarded_cards", [])],
        ai_settings={"aiType": "mccfr"}
    )

    timeout_event = Event()
    result = {"move": None}
    Thread(target=cfr_agent.get_move, args=(game_state, timeout_event, result)).start().join(timeout=5)

    move = result["move"]
    if move and "error" not in move:
        await save_progress_async()
        next_state = game_state.apply_action(move)
        if next_state.is_terminal():
            royalties = next_state.calculate_royalties()
            return jsonify({
                "move": serialize_move(move),
                "royalties": royalties,
                "total_royalty": sum(royalties.values()) if isinstance(royalties, dict) else 0,
                "game_over": True
            })
        return jsonify({"move": serialize_move(move)})
    return jsonify({"error": "Ошибка хода ИИ"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=10000)
