import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
from collections import defaultdict, Counter
import random
import itertools
from threading import Event, Thread
import time
import math
import logging
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class Card:
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    SUITS = ["♥", "♦", "♣", "♠"]

    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS or suit not in self.SUITS:
            raise ValueError(f"Invalid card: {rank}{suit}")
        self.rank = rank
        self.suit = suit

    def __repr__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __eq__(self, other: Union["Card", Dict]) -> bool:
        if isinstance(other, dict):
            return self.rank == other.get("rank") and self.suit == other.get("suit")
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def to_dict(self) -> Dict[str, str]:
        return {"rank": self.rank, "suit": self.suit}

    @staticmethod
    def from_dict(card_dict: Dict[str, str]) -> "Card":
        return Card(card_dict["rank"], card_dict["suit"])

    @staticmethod
    def get_all_cards() -> List["Card"]:
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]


class Hand:
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards = cards if cards is not None else []

    def add_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("Card must be an instance of Card")
        self.cards.append(card)

    def remove_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("Card must be an instance of Card")
        self.cards.remove(card)

    def __repr__(self) -> str:
        return ", ".join(map(str, self.cards))

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self):
        return iter(self.cards)

    def __getitem__(self, index: int) -> Card:
        return self.cards[index]


class Board:
    def __init__(self):
        self.top: List[Card] = []
        self.middle: List[Card] = []
        self.bottom: List[Card] = []

    def place_card(self, line: str, card: Card) -> None:
        if line == "top" and len(self.top) < 3:
            self.top.append(card)
        elif line == "middle" and len(self.middle) < 5:
            self.middle.append(card)
        elif line == "bottom" and len(self.bottom) < 5:
            self.bottom.append(card)
        else:
            raise ValueError(f"Cannot place card in {line}")

    def is_full(self) -> bool:
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self) -> None:
        self.top, self.middle, self.bottom = [], [], []

    def __repr__(self) -> str:
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def get_cards(self, line: str) -> List[Card]:
        return getattr(self, line, [])


class GameState:
    def __init__(
        self,
        selected_cards: Optional[List[Card]] = None,
        board: Optional[Board] = None,
        discarded_cards: Optional[List[Card]] = None,
        ai_settings: Optional[Dict] = None,
        deck: Optional[List[Card]] = None,
    ):
        self.selected_cards = Hand(selected_cards) if selected_cards else Hand()
        self.board = board if board else Board()
        self.discarded_cards = discarded_cards if discarded_cards else []
        self.ai_settings = ai_settings if ai_settings else {}
        self.current_player = 0
        self.deck = deck if deck else Card.get_all_cards()
        self.rank_map = {r: i for i, r in enumerate(Card.RANKS)}
        self.suit_map = {s: i for i, s in enumerate(Card.SUITS)}

    def get_current_player(self) -> int:
        return self.current_player

    def is_terminal(self) -> bool:
        return self.board.is_full()

    def get_available_cards(self) -> List[Card]:
        used = set(self.discarded_cards + sum([self.board.get_cards(line) for line in ["top", "middle", "bottom"]], []))
        return [card for card in self.deck if card not in used]

    def get_actions(self) -> List[Dict[str, List[Card]]]:
        if self.is_terminal():
            return []

        num_cards = len(self.selected_cards)
        free_slots = {
            "top": 3 - len(self.board.top),
            "middle": 5 - len(self.board.middle),
            "bottom": 5 - len(self.board.bottom)
        }
        total_free = sum(free_slots.values())
        actions = []

        if num_cards == 3:
            for i in range(3):
                remaining = [c for j, c in enumerate(self.selected_cards.cards) if j != i]
                for line in ["top", "middle", "bottom"]:
                    if free_slots[line] >= 2:
                        actions.append({
                            "top": remaining if line == "top" else [],
                            "middle": remaining if line == "middle" else [],
                            "bottom": remaining if line == "bottom" else [],
                            "discarded": [self.selected_cards.cards[i]]
                        })
        elif self.ai_settings.get("fantasyMode", False) and num_cards > 3:
            for perm in itertools.permutations(self.selected_cards.cards):
                action = {
                    "top": list(perm[:3]),
                    "middle": list(perm[3:8]),
                    "bottom": list(perm[8:13]),
                    "discarded": list(perm[13:])
                }
                if self.is_valid_fantasy_repeat(action):
                    actions.append(action)
        elif num_cards > total_free:
            for cards_to_place in itertools.combinations(self.selected_cards.cards, total_free):
                discarded = [c for c in self.selected_cards.cards if c not in cards_to_place]
                for perm in itertools.permutations(cards_to_place):
                    action = {
                        "top": list(perm[:free_slots["top"]]),
                        "middle": list(perm[free_slots["top"]:free_slots["top"] + free_slots["middle"]]),
                        "bottom": list(perm[free_slots["top"] + free_slots["middle"]:]),
                        "discarded": discarded
                    }
                    if not self.is_dead_hand_after_action(action):
                        actions.append(action)
        else:
            for perm in itertools.permutations(self.selected_cards.cards):
                action = {
                    "top": list(perm[:free_slots["top"]]),
                    "middle": list(perm[free_slots["top"]:free_slots["top"] + free_slots["middle"]]),
                    "bottom": list(perm[free_slots["top"] + free_slots["middle"]:]),
                    "discarded": []
                }
                if not self.is_dead_hand_after_action(action):
                    actions.append(action)
        return actions

    def is_dead_hand_after_action(self, action: Dict[str, List[Card]]) -> bool:
        new_board = Board()
        for line in ["top", "middle", "bottom"]:
            new_board.__setattr__(line, self.board.get_cards(line) + action.get(line, []))
        temp_state = GameState(board=new_board)
        return temp_state.is_dead_hand()

    def is_valid_fantasy_repeat(self, action: Dict[str, List[Card]]) -> bool:
        new_board = Board()
        for line in ["top", "middle", "bottom"]:
            new_board.__setattr__(line, self.board.get_cards(line) + action.get(line, []))
        temp_state = GameState(board=new_board)
        if temp_state.is_dead_hand():
            return False
        top_rank, _ = temp_state.evaluate_hand(new_board.top)
        bottom_rank, _ = temp_state.evaluate_hand(new_board.bottom)
        return top_rank == 7 or bottom_rank <= 3

    def apply_action(self, action: Dict[str, List[Card]]) -> "GameState":
        new_board = Board()
        for line in ["top", "middle", "bottom"]:
            new_board.__setattr__(line, self.board.get_cards(line) + action.get(line, []))
        new_discarded = self.discarded_cards + (action.get("discarded", []) if isinstance(action.get("discarded"), list) else [action.get("discarded")])
        return GameState(board=new_board, discarded_cards=new_discarded, ai_settings=self.ai_settings, deck=self.deck[:])

    def get_information_set(self) -> str:
        def sort_cards(cards): return sorted(cards, key=lambda c: (self.rank_map[c.rank], self.suit_map[c.suit]))
        return f"T:{','.join(map(str, sort_cards(self.board.top)))}|M:{','.join(map(str, sort_cards(self.board.middle)))}|B:{','.join(map(str, sort_cards(self.board.bottom)))}|D:{','.join(map(str, sort_cards(self.discarded_cards)))}|S:{','.join(map(str, sort_cards(self.selected_cards.cards)))}"

    def get_payoff(self) -> Dict[str, int]:
        if not self.is_terminal():
            raise ValueError("Game is not terminal")
        royalties = self.calculate_royalties()
        return {"total": -1000} if royalties == "Фол" else {"total": sum(royalties.values())}

    def is_dead_hand(self) -> bool:
        if not self.board.is_full():
            return False
        top_rank, _ = self.evaluate_hand(self.board.top)
        middle_rank, _ = self.evaluate_hand(self.board.middle)
        bottom_rank, _ = self.evaluate_hand(self.board.bottom)
        return top_rank > middle_rank or middle_rank > bottom_rank

    def calculate_royalties(self) -> Dict[str, Union[int, str]]:
        """
        Вычисляет роялти для текущего состояния.

        Returns:
            Dict[str, Union[int, str]]: Роялти или "Фол".
        """
        if self.is_dead_hand():
            return "Фол"
        return {line: self.get_line_royalties(line) for line in ["top", "middle", "bottom"]}

    def get_line_royalties(self, line: str) -> int:
        """
        Вычисляет роялти для линии по правилам OFC.

        Args:
            line (str): Линия ("top", "middle", "bottom").

        Returns:
            int: Очки роялти.
        """
        cards = getattr(self.board, line)
        if not cards:
            return 0
        rank, _ = self.evaluate_hand(cards)
        rank_map = {r: i for i, r in enumerate(Card.RANKS[::-1])}
        if line == "top":
            if rank == 7:  # Сет
                return 10 + rank_map[cards[0].rank]
            elif rank == 8:  # Пара
                pair_rank = [c.rank for c in cards if cards.count(c) == 2][0]
                idx = rank_map[pair_rank]
                return max(0, idx - 6) + 1 if idx <= 8 else 0
        elif line == "middle":
            return {1: 50, 2: 30, 3: 20, 4: 12, 5: 8, 6: 4, 7: 2}.get(rank, 0)
        elif line == "bottom":
            return {1: 25, 2: 15, 3: 10, 4: 6, 5: 4, 6: 2}.get(rank, 0)
        return 0

    def evaluate_hand(self, cards: List[Card]) -> Tuple[int, float]:
        """
        Оценивает комбинацию карт.

        Args:
            cards (List[Card]): Карты в линии.

        Returns:
            Tuple[int, float]: Ранг (меньше — сильнее) и оценка.
        """
        if not cards or len(cards) not in (3, 5):
            return 11, 0
        ranks = [self.rank_map[c.rank] for c in cards]
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        sorted_ranks = sorted(ranks)

        if len(cards) == 5:
            is_flush = len(set(suits)) == 1
            is_straight = (sorted_ranks == list(range(min(ranks), min(ranks) + 5))) or sorted_ranks == [0, 1, 2, 3, 12]
            if is_flush and sorted_ranks == [8, 9, 10, 11, 12]:
                return 1, 25  # Royal Flush
            if is_flush and is_straight:
                return 2, 15  # Straight Flush
            if 4 in rank_counts.values():
                return 3, 10 + max(k for k, v in rank_counts.items() if v == 4) / 100  # Four of a Kind
            if 3 in rank_counts.values() and 2 in rank_counts.values():
                return 4, 6 + max(k for k, v in rank_counts.items() if v == 3) / 100  # Full House
            if is_flush:
                return 5, 4 + sum(ranks) / 1000  # Flush
            if is_straight:
                return 6, 2 + sum(ranks) / 1000  # Straight
        if 3 in rank_counts.values():
            return 7, 2 + max(k for k, v in rank_counts.items() if v == 3) / 100  # Three of a Kind
        if list(rank_counts.values()).count(2) == 2:
            return 8, sum(k for k, v in rank_counts.items() if v == 2) / 1000  # Two Pair
        if 2 in rank_counts.values():
            return 9, max(k for k, v in rank_counts.items() if v == 2) / 1000  # One Pair
        return 10, max(ranks) / 10000  # High Card


class CFRNode:
    def __init__(self, actions: List[Dict[str, List[Card]]]):
        self.actions = actions
        self.regret_sum = np.zeros(len(actions), dtype=np.float32)
        self.strategy_sum = np.zeros(len(actions), dtype=np.float32)

    @jit
    def get_strategy(self, realization_weight: float) -> np.ndarray:
        strategy = jnp.maximum(self.regret_sum, 0)
        normalizing_sum = jnp.sum(strategy)
        strategy = jax.lax.cond(
            normalizing_sum > 0,
            lambda s: s / normalizing_sum,
            lambda s: jnp.ones_like(s) / len(s),
            strategy
        )
        self.strategy_sum += realization_weight * strategy
        return strategy

    @jit
    def get_average_strategy(self) -> np.ndarray:
        normalizing_sum = jnp.sum(self.strategy_sum)
        return jax.lax.cond(
            normalizing_sum > 0,
            lambda s: s / normalizing_sum,
            lambda s: jnp.ones_like(s) / len(s),
            self.strategy_sum
        )


class CFRAgent:
    def __init__(self, iterations: int = 500000, stop_threshold: float = 0.0001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 100

    @jit
    def cfr(self, game_state: GameState, p0: float, p1: float, iteration: int) -> float:
        if game_state.is_terminal():
            payoff = game_state.get_payoff()
            return payoff["total"] if isinstance(payoff, dict) else payoff

        info_set = game_state.get_information_set()
        if info_set not in self.nodes:
            self.nodes[info_set] = CFRNode(game_state.get_actions())
        node = self.nodes[info_set]

        strategy = node.get_strategy(p0 if game_state.get_current_player() == 0 else p1)
        util = np.zeros(len(node.actions), dtype=np.float32)

        def body_fun(i, u):
            next_state = game_state.apply_action(node.actions[i])
            u = u.at[i].set(-self.cfr(next_state, p0 * strategy[i], p1, iteration) if game_state.get_current_player() == 0
                           else -self.cfr(next_state, p0, p1 * strategy[i], iteration))
            return u

        util = jax.lax.fori_loop(0, len(node.actions), body_fun, util)
        node_util = jnp.dot(strategy, util)

        def update_regrets(player):
            regret = util - node_util
            self.nodes[info_set].regret_sum += (p1 if player == 0 else p0) * regret

        jax.lax.cond(game_state.get_current_player() == 0, lambda: update_regrets(0), lambda: update_regrets(1))
        return node_util

    def train(self, timeout_event: Event, result: Dict) -> None:
        for i in range(self.iterations):
            if timeout_event.is_set():
                logger.info(f"Training interrupted at {i} iterations.")
                break
            all_cards = Card.get_all_cards()
            np.random.shuffle(all_cards)
            game_state = GameState(deck=all_cards[:])
            game_state.selected_cards = Hand(all_cards[:5])
            self.cfr(game_state, 1.0, 1.0, i + 1)
            if (i + 1) % self.save_interval == 0:
                self.save_progress()
                if self.check_convergence():
                    logger.info(f"Converged at {i + 1} iterations.")
                    break

    def check_convergence(self) -> bool:
        for node in self.nodes.values():
            avg_strategy = node.get_average_strategy()
            if any(abs(p - 1.0 / len(node.actions)) > self.stop_threshold for p in avg_strategy):
                return False
        return True

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        actions = game_state.get_actions()
        if not actions:
            result["move"] = {"error": "Нет доступных ходов"}
            return

        info_set = game_state.get_information_set()
        if info_set in self.nodes:
            strategy = self.nodes[info_set].get_average_strategy()
            best_move = self.nodes[info_set].actions[np.argmax(strategy)]
        else:
            best_value, best_move = float("-inf"), None
            for action in actions:
                value = self.shallow_search(game_state.apply_action(action), 2, timeout_event)
                if value > best_value:
                    best_value, best_move = value, action
            self.cfr(game_state, 1.0, 1.0, 1)  # Обучение на новом состоянии
        result["move"] = best_move

    def shallow_search(self, state: GameState, depth: int, timeout_event: Event) -> float:
        if depth == 0 or state.is_terminal() or timeout_event.is_set():
            return self.baseline_evaluation(state)
        best_value = float("-inf")
        for action in state.get_actions():
            if timeout_event.is_set():
                return 0
            value = -self.shallow_search(state.apply_action(action), depth - 1, timeout_event)
            best_value = max(best_value, value)
        return best_value

    def baseline_evaluation(self, state: GameState) -> float:
        if state.is_dead_hand():
            return -1000
        royalties = state.calculate_royalties()
        if royalties == "Фол":
            return -1000
        return sum(royalties.values()) + sum(
            self.calculate_potential(getattr(state.board, line), line, state.board, state.get_available_cards())
            for line in ["top", "middle", "bottom"]
        )

    def calculate_potential(self, cards: List[Card], line: str, board: Board, available_cards: List[Card]) -> float:
        if not cards or (line == "top" and len(cards) == 3) or (line in ["middle", "bottom"] and len(cards) == 5):
            return 0
        potential = 0
        suits = Counter(c.suit for c in cards)
        ranks = [self.rank_map[c.rank] for c in cards]
        if line != "top" and max(suits.values(), default=0) >= 2:
            potential += 0.7 * (5 - len(cards)) / 3
        if line != "top" and len(set(ranks)) >= 2:
            potential += 0.5 * (5 - len(cards)) / 3
        if line == "top" and len(cards) == 2 and cards[0].rank == cards[1].rank:
            potential += 0.3
        return potential

    def save_progress(self) -> None:
        import utils
        data = {"nodes": self.nodes, "iterations": self.iterations, "stop_threshold": self.stop_threshold}
        utils.save_ai_progress(data, "cfr_data.pkl")

    def load_progress(self) -> None:
        import utils
        data = utils.load_ai_progress("cfr_data.pkl")
        if data:
            self.nodes = data["nodes"]
            self.iterations = data["iterations"]
            self.stop_threshold = data.get("stop_threshold", 0.0001)


class RandomAgent:
    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        actions = game_state.get_actions()
        result["move"] = random.choice(actions) if actions else {"error": "Нет доступных ходов"}
