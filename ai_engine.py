import random
import itertools
from collections import defaultdict, Counter
import utils  # Предполагается, что здесь функции save_ai_progress и load_ai_progress
from threading import Event, Thread
import time
import math
import logging
from typing import List, Dict, Tuple, Optional, Union

# Настройка логирования
logger = logging.getLogger(__name__)


class Card:
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    SUITS = ["♥", "♦", "♣", "♠"]

    def __init__(self, rank: str, suit: str):
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank}. Rank must be one of: {self.RANKS}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit}. Suit must be one of: {self.SUITS}")
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
            raise TypeError("card must be an instance of Card")
        self.cards.append(card)

    def remove_card(self, card: Card) -> None:
        if not isinstance(card, Card):
            raise TypeError("card must be an instance of Card")
        try:
            self.cards.remove(card)
        except ValueError:
            logger.warning(f"Card {card} not found in hand: {self.cards}")

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
        if line == "top":
            if len(self.top) >= 3:
                raise ValueError("Top line is full")
            self.top.append(card)
        elif line == "middle":
            if len(self.middle) >= 5:
                raise ValueError("Middle line is full")
            self.middle.append(card)
        elif line == "bottom":
            if len(self.bottom) >= 5:
                raise ValueError("Bottom line is full")
            self.bottom.append(card)
        else:
            raise ValueError(f"Invalid line: {line}. Line must be one of: 'top', 'middle', 'bottom'")

    def is_full(self) -> bool:
        return len(self.top) == 3 and len(self.middle) == 5 and len(self.bottom) == 5

    def clear(self) -> None:
        self.top = []
        self.middle = []
        self.bottom = []

    def __repr__(self) -> str:
        return f"Top: {self.top}\nMiddle: {self.middle}\nBottom: {self.bottom}"

    def get_cards(self, line: str) -> List[Card]:
        if line == "top":
            return self.top
        elif line == "middle":
            return self.middle
        elif line == "bottom":
            return self.bottom
        else:
            raise ValueError("Invalid line specified")


class GameState:
    def __init__(
        self,
        selected_cards: Optional[List[Card]] = None,
        board: Optional[Board] = None,
        discarded_cards: Optional[List[Card]] = None,
        ai_settings: Optional[Dict] = None,
        deck: Optional[List[Card]] = None,
    ):
        self.selected_cards: Hand = Hand(selected_cards) if selected_cards is not None else Hand()
        self.board: Board = board if board is not None else Board()
        self.discarded_cards: List[Card] = discarded_cards if discarded_cards is not None else []
        self.ai_settings: Dict = ai_settings if ai_settings is not None else {}
        self.current_player: int = 0
        self.deck: List[Card] = deck if deck is not None else self.create_deck()
        self.rank_map: Dict[str, int] = {rank: i for i, rank in enumerate(Card.RANKS)}
        self.suit_map: Dict[str, int] = {suit: i for i, suit in enumerate(Card.SUITS)}

    def create_deck(self) -> List[Card]:
        """Creates a standard deck of 52 cards."""
        return [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]

    def get_current_player(self) -> int:
        return self.current_player

    def is_terminal(self) -> bool:
        """Checks if the game is in a terminal state (all lines are full)."""
        return self.board.is_full()

    def get_num_cards_to_draw(self) -> int:
        """Returns the number of cards to draw based on the current game state."""
        placed_cards = sum(len(row) for row in [self.board.top, self.board.middle, self.board.bottom])
        if placed_cards == 5:
            return 3
        elif placed_cards in (7, 10):
            return 3
        elif placed_cards >= 13:
            return 0
        return 0

    def get_available_cards(self) -> List[Card]:
        """Returns a list of cards that are still available in the deck."""
        used_cards = set(self.discarded_cards)
        available_cards = [card for card in self.deck if card not in used_cards]
        return available_cards

    def get_actions(self) -> List[Dict[str, List[Card]]]:
        """Returns the valid actions for the current state."""
        logger.debug("get_actions - START")
        if self.is_terminal():
            logger.debug("get_actions - Game is terminal, returning empty actions")
            return []

        num_cards = len(self.selected_cards)
        actions = []
        used_cards = set()
        for line in [self.board.top, self.board.middle, self.board.bottom]:
            used_cards.update([card for card in line if card is not None])
        used_cards.update(self.discarded_cards)
        
        # Определяем, какие слоты свободны
        free_slots = {
            "top": [i for i in range(3) if i >= len(self.board.top)],
            "middle": [i for i in range(5) if i >= len(self.board.middle)],
            "bottom": [i for i in range(5) if i >= len(self.board.bottom)]
        }
        
        total_free_slots = len(free_slots["top"]) + len(free_slots["middle"]) + len(free_slots["bottom"])

        if num_cards > 0:
            try:
                # Режим фантазии
                if self.ai_settings.get("fantasyMode", False):
                    valid_fantasy_repeats = []
                    for p in itertools.permutations(self.selected_cards.cards):
                        action = {
                            "top": list(p[:3]),
                            "middle": list(p[3:8]),
                            "bottom": list(p[8:13]),
                            "discarded": list(p[13:]),
                        }
                        if self.is_valid_fantasy_repeat(action):
                            valid_fantasy_repeats.append(action)
                    if valid_fantasy_repeats:
                        actions = sorted(
                            valid_fantasy_repeats,
                            key=lambda a: self.calculate_action_royalty(a),
                            reverse=True,
                        )
                    else:
                        actions = sorted(
                            [
                                {
                                    "top": list(p[:3]),
                                    "middle": list(p[3:8]),
                                    "bottom": list(p[8:13]),
                                    "discarded": list(p[13:]),
                                }
                                for p in itertools.permutations(self.selected_cards.cards)
                            ],
                            key=lambda a: self.calculate_action_royalty(a),
                            reverse=True,
                        )
                
                # Особый случай: ровно 3 карты - всегда размещаем 2, сбрасываем 1
                elif num_cards == 3:
                    for discarded_index in range(3):
                        remaining_cards = [
                            card for i, card in enumerate(self.selected_cards.cards) if i != discarded_index
                        ]
                        
                        # Генерируем все возможные размещения 2 карт по свободным слотам
                        for placement in self._generate_placements_for_free_slots(remaining_cards, free_slots):
                            action = {
                                "top": placement["top"],
                                "middle": placement["middle"],
                                "bottom": placement["bottom"],
                                "discarded": [self.selected_cards.cards[discarded_index]]
                            }
                            actions.append(action)
                
                # Общий случай
                else:
                    remaining_cards = list(self.selected_cards.cards)
                    
                    # Если у нас больше карт, чем свободных слотов, нужно выбрать, какие разместить
                    if num_cards > total_free_slots:
                        for cards_to_place in itertools.combinations(remaining_cards, total_free_slots):
                            cards_to_discard = [card for card in remaining_cards if card not in cards_to_place]
                            
                            # Генерируем все возможные размещения выбранных карт по свободным слотам
                            for placement in self._generate_placements_for_free_slots(list(cards_to_place), free_slots):
                                action = {
                                    "top": placement["top"],
                                    "middle": placement["middle"],
                                    "bottom": placement["bottom"],
                                    "discarded": cards_to_discard
                                }
                                actions.append(action)
                    
                    # Если у нас ровно столько карт, сколько свободных слотов, или меньше
                    else:
                        for placement in self._generate_placements_for_free_slots(remaining_cards, free_slots):
                            action = {
                                "top": placement["top"],
                                "middle": placement["middle"],
                                "bottom": placement["bottom"],
                                "discarded": []
                            }
                            actions.append(action)
            
            except Exception as e:
                logger.exception(f"Error in get_actions: {e}")
                return []
        
        logger.debug(f"Generated {len(actions)} actions")
        logger.debug("get_actions - END")
        return actions

    def _generate_placements_for_free_slots(self, cards: List[Card], free_slots: Dict[str, List[int]]) -> List[Dict[str, List[Card]]]:
        """Generates all valid placements of cards in free slots."""
        if not cards:
            return [{"top": [], "middle": [], "bottom": []}]
        
        placements = []
        
        # Рекурсивно генерируем все возможные размещения карт по свободным слотам
        def backtrack(index: int, current_placement: Dict[str, List[Card]]):
            if index == len(cards):
                # Проверяем, что размещение соответствует правилам игры
                temp_board = Board()
                temp_board.top = self.board.top + current_placement["top"]
                temp_board.middle = self.board.middle + current_placement["middle"]
                temp_board.bottom = self.board.bottom + current_placement["bottom"]
                
                temp_state = GameState(board=temp_board, ai_settings=self.ai_settings)
                if not temp_state.is_dead_hand():
                    placements.append({
                        "top": current_placement["top"][:],
                        "middle": current_placement["middle"][:],
                        "bottom": current_placement["bottom"][:]
                    })
                return
            
            # Пробуем разместить текущую карту в каждой линии
            for line in ["top", "middle", "bottom"]:
                if len(current_placement[line]) < len(free_slots[line]):
                    current_placement[line].append(cards[index])
                    backtrack(index + 1, current_placement)
                    current_placement[line].pop()
        
        backtrack(0, {"top": [], "middle": [], "bottom": []})
        return placements

    def is_valid_fantasy_entry(self, action: Dict[str, List[Card]]) -> bool:
        """Checks if an action leads to a valid fantasy mode entry."""
        new_board = Board()
        new_board.top = self.board.top + action.get("top", [])
        new_board.middle = self.board.middle + action.get("middle", [])
        new_board.bottom = self.board.bottom + action.get("bottom", [])

        temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(new_board.top)
        return top_rank <= 8 and new_board.top[0].rank in ["Q", "K", "A"]

    def is_valid_fantasy_repeat(self, action: Dict[str, List[Card]]) -> bool:
        """Checks if an action leads to a valid fantasy mode repeat."""
        new_board = Board()
        new_board.top = self.board.top + action.get("top", [])
        new_board.middle = self.board.middle + action.get("middle", [])
        new_board.bottom = self.board.bottom + action.get("bottom", [])

        temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(new_board.top)
        bottom_rank, _ = temp_state.evaluate_hand(new_board.bottom)

        if top_rank == 7:
            return True
        if bottom_rank <= 3:
            return True

        return False

    def calculate_action_royalty(self, action: Dict[str, List[Card]]) -> Dict[str, int]:
        """Calculates the royalty for a given action."""
        new_board = Board()
        new_board.top = self.board.top + action.get("top", [])
        new_board.middle = self.board.middle + action.get("middle", [])
        new_board.bottom = self.board.bottom + action.get("bottom", [])

        temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
        return temp_state.calculate_royalties()

    def apply_action(self, action: Dict[str, List[Card]]) -> "GameState":
        """Applies an action to the current state and returns the new state."""
        new_board = Board()
        new_board.top = self.board.top + action.get("top", [])
        new_board.middle = self.board.middle + action.get("middle", [])
        new_board.bottom = self.board.bottom + action.get("bottom", [])

        new_discarded_cards = self.discarded_cards[:]
        if "discarded" in action and action["discarded"]:
            if isinstance(action["discarded"], list):
                for card in action["discarded"]:
                    self.mark_card_as_used(card)
            else:
                self.mark_card_as_used(action["discarded"])

        for line in ["top", "middle", "bottom"]:
            for card in action.get(line, []):
                self.mark_card_as_used(card)

        new_game_state = GameState(
            selected_cards=Hand(),
            board=new_board,
            discarded_cards=new_discarded_cards,
            ai_settings=self.ai_settings,
            deck=self.deck[:],
        )

        return new_game_state

    def get_information_set(self) -> str:
        """Returns a string representation of the current information set."""
        def card_to_string(card: Card) -> str:
            return str(card)

        def sort_cards(cards: List[Card]) -> List[Card]:
            return sorted(cards, key=lambda card: (self.rank_map[card.rank], self.suit_map[card.suit]))

        top_str = ",".join(map(card_to_string, sort_cards(self.board.top)))
        middle_str = ",".join(map(card_to_string, sort_cards(self.board.middle)))
        bottom_str = ",".join(map(card_to_string, sort_cards(self.board.bottom)))
        discarded_str = ",".join(map(card_to_string, sort_cards(self.discarded_cards)))
        selected_str = ",".join(map(card_to_string, sort_cards(self.selected_cards.cards)))

        return f"T:{top_str}|M:{middle_str}|B:{bottom_str}|D:{discarded_str}|S:{selected_str}"

    def get_payoff(self) -> Dict[str, int]:
        """Calculates the payoff for the current state."""
        if not self.is_terminal():
            raise ValueError("Game is not in a terminal state")

        if self.is_dead_hand():
            return -self.calculate_royalties()

        return self.calculate_royalties()

    def is_dead_hand(self) -> bool:
        """Checks if the hand is a dead hand (invalid combination order)."""
        if not self.board.is_full():
            return False

        top_rank, _ = self.evaluate_hand(self.board.top)
        middle_rank, _ = self.evaluate_hand(self.board.middle)
        bottom_rank, _ = self.evaluate_hand(self.board.bottom)

        return top_rank > middle_rank or middle_rank > bottom_rank

    def get_line_royalties(self, line: str) -> int:
        """Calculates royalties for a specific line."""
        cards = getattr(self.board, line)
        if not cards:
            return 0

        rank, _ = self.evaluate_hand(cards)
        if line == "top":
            if rank == 7:
                return 10 + Card.RANKS.index(cards[0].rank)
            elif rank == 8:
                return self.get_pair_bonus(cards)
            elif rank == 9:
                return self.get_high_card_bonus(cards)
        elif line == "middle":
            if rank <= 6:
                return self.get_royalties_for_hand(rank) * 2
        elif line == "bottom":
            if rank <= 6:
                return self.get_royalties_for_hand(rank)
        return 0

    def calculate_royalties(self) -> Dict[str, int]:
        """Calculates royalties for the current state based on the rules."""
        if self.is_dead_hand():
            return 0

        royalties = {}
        lines = {"top": self.board.top, "middle": self.board.middle, "bottom": self.board.bottom}

        for line_name, cards in lines.items():
            royalties[line_name] = self.get_line_royalties(line_name)

        return royalties

    def get_royalties_for_hand(self, hand_rank: int) -> int:
        if hand_rank == 1:
            return 25
        elif hand_rank == 2:
            return 15
        elif hand_rank == 3:
            return 10
        elif hand_rank == 4:
            return 6
        elif hand_rank == 5:
            return 4
        elif hand_rank == 6:
            return 2
        return 0

    def get_line_score(self, line: str, cards: List[Card]) -> int:
        """Calculates the score for a specific line based on hand rankings."""
        if not cards:
            return 0

        rank, score = self.evaluate_hand(cards)
        return score

    def get_pair_bonus(self, cards: List[Card]) -> int:
        """Calculates the bonus for a pair in the top line."""
        if len(cards) != 3:
            return 0
        ranks = [card.rank for card in cards]
        for rank in Card.RANKS[::-1]:
            if ranks.count(rank) == 2:
                return 1 + Card.RANKS.index(rank) - Card.RANKS.index("6") if rank >= "6" else 0

    def get_high_card_bonus(self, cards: List[Card]) -> int:
        """Calculates the bonus for a high card in the top line."""
        if len(cards) != 3 or not all(isinstance(card, Card) for card in cards):
            return 0
        ranks = [card.rank for card in cards]
        if len(set(ranks)) == 3:
            high_card = max(ranks, key=Card.RANKS.index)
            return 1 if high_card == "A" else 0

    def mark_card_as_used(self, card: Card) -> None:
        """Marks a card as used (either placed on the board or discarded)."""
        if card not in self.discarded_cards:
            self.discarded_cards.append(card)

    def get_fantasy_bonus(self) -> int:
        """Calculates the bonus for fantasy mode."""
        bonus = 0
        top_rank, _ = self.evaluate_hand(self.board.top)

        if top_rank <= 8 and self.board.top[0].rank in ["Q", "K", "A"]:
            if self.ai_settings.get("fantasyType") == "progressive":
                if self.board.top[0].rank == "Q":
                    bonus += 14  # 14 cards for QQ
                elif self.board.top[0].rank == "K":
                    bonus += 15  # 15 cards for KK
                elif self.board.top[0].rank == "A":
                    bonus += 16  # 16 cards for AA
                elif top_rank == 7:  # Set
                    bonus += 17  # 17 cards for set from 222 to AAA
            else:  # Normal fantasy
                bonus += 14  # 14 cards

            if self.is_fantasy_repeat(action={}):
                bonus += 14  # Fantasy repeat - 14 cards (regardless of type)

        return bonus

    def is_fantasy_repeat(self, action: Dict[str, List[Card]]) -> bool:
        """Checks if the conditions for fantasy repeat are met."""
        new_board = Board()
        new_board.top = self.board.top + action.get("top", [])
        new_board.middle = self.board.middle + action.get("middle", [])
        new_board.bottom = self.board.bottom + action.get("bottom", [])

        temp_state = GameState(board=new_board, ai_settings=self.ai_settings)
        if temp_state.is_dead_hand():
            return False

        top_rank, _ = temp_state.evaluate_hand(new_board.top)
        bottom_rank, _ = temp_state.evaluate_hand(new_board.bottom)

        if top_rank == 7:  # Set in top row
            return True
        if bottom_rank <= 3:  # Four of a Kind or better in bottom row
            return True

        return False

    def evaluate_hand(self, cards: List[Card]) -> Tuple[int, float]:
        """Evaluates the hand and returns a rank (lower is better) and a score."""
        if not cards or not all(isinstance(card, Card) for card in cards):
            return 11, 0  # Return a low rank for invalid hands

        n = len(cards)
        if n == 5:
            if self.is_royal_flush(cards):
                return 1, 25
            if self.is_straight_flush(cards):
                return 2, 15
            if self.is_four_of_a_kind(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 4][0]
                return 3, 10 + Card.RANKS.index(rank) / 100
            if self.is_full_house(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 3][0]
                return 4, 6 + Card.RANKS.index(rank) / 100
            if self.is_flush(cards):
                score = 4 + sum(Card.RANKS.index(card.rank) for card in cards) / 1000
                return 5, score
            if self.is_straight(cards):
                score = 2 + sum(Card.RANKS.index(card.rank) for card in cards) / 1000
                return 6, score
            if self.is_three_of_a_kind(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 3][0]
                return 7, 2 + Card.RANKS.index(rank) / 100
            if self.is_two_pair(cards):
                ranks = sorted(
                    [Card.RANKS.index(card.rank) for card in cards if [card.rank for card in cards].count(card.rank) == 2],
                    reverse=True,
                )
                return 8, sum(ranks) / 1000
            if self.is_one_pair(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 2][0]
                return 9, Card.RANKS.index(rank) / 1000
            score = sum(Card.RANKS.index(card.rank) for card in cards) / 10000
            return 10, score

        elif n == 3:
            if self.is_three_of_a_kind(cards):
                rank = cards[0].rank
                return 7, 10 + Card.RANKS.index(rank)
            if self.is_one_pair(cards):
                rank = [card.rank for card in cards if [card.rank for card in cards].count(card.rank) == 2][0]
                return 8, self.get_pair_bonus(cards)
            return 9, self.get_high_card_bonus(cards)

        else:
            return 11, 0

    def is_royal_flush(self, cards: List[Card]) -> bool:
        if not self.is_flush(cards):
            return False
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        return ranks == [8, 9, 10, 11, 12]

    def is_straight_flush(self, cards: List[Card]) -> bool:
        return self.is_straight(cards) and self.is_flush(cards)

    def is_four_of_a_kind(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 4 for r in ranks)

    def is_full_house(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks) and any(ranks.count(r) == 2 for r in ranks)

    def is_flush(self, cards: List[Card]) -> bool:
        suits = [card.suit for card in cards]
        return len(set(suits)) == 1

    def is_straight(self, cards: List[Card]) -> bool:
        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        if ranks == [0, 1, 2, 3, 12]:
            return True
        return all(ranks[i + 1] - ranks[i] == 1 for i in range(len(ranks) - 1))

    def is_three_of_a_kind(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 3 for r in ranks)

    def is_two_pair(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        pairs = [r for r in set(ranks) if ranks.count(r) == 2]
        return len(pairs) == 2

    def is_one_pair(self, cards: List[Card]) -> bool:
        ranks = [card.rank for card in cards]
        return any(ranks.count(r) == 2 for r in ranks)


class CFRNode:
    def __init__(self, actions: List[Dict[str, List[Card]]]):
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        self.actions = actions

    def get_strategy(self, realization_weight: float) -> Dict[Dict[str, List[Card]], float]:
        normalizing_sum = 0
        strategy = defaultdict(float)
        for a in self.actions:
            strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
            normalizing_sum += strategy[a]

        for a in self.actions:
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0 / len(self.actions)
            self.strategy_sum[a] += realization_weight * strategy[a]
        return strategy

    def get_average_strategy(self) -> Dict[Dict[str, List[Card]], float]:
        avg_strategy = defaultdict(float)
        normalizing_sum = sum(self.strategy_sum.values())
        if normalizing_sum > 0:
            for a in self.actions:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
        else:
            for a in self.actions:
                avg_strategy[a] = 1.0 / len(self.actions)
        return avg_strategy


class CFRAgent:
    def __init__(self, iterations: int = 1000, stop_threshold: float = 0.001):
        self.nodes = {}
        self.iterations = iterations
        self.stop_threshold = stop_threshold
        self.save_interval = 100

    def cfr(
        self,
        game_state: GameState,
        p0: float,
        p1: float,
        timeout_event: Event,
        result: Dict,
        iteration: int,
    ) -> float:
        if timeout_event.is_set():
            logger.info("CFR timed out!")
            return 0

        if game_state.is_terminal():
            payoff = game_state.get_payoff()
            logger.debug(f"cfr called in terminal state. Payoff: {payoff}")
            return payoff

        player = game_state.get_current_player()
        info_set = game_state.get_information_set()
        logger.debug(f"cfr called for info_set: {info_set}, player: {player}")

        if info_set not in self.nodes:
            actions = game_state.get_actions()
            if not actions:
                logger.debug("No actions available for this state.")
                return 0
            self.nodes[info_set] = CFRNode(actions)
        node = self.nodes[info_set]

        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = defaultdict(float)
        node_util = 0

        for a in node.actions:
            if timeout_event.is_set():
                logger.info("CFR timed out during action loop!")
                return 0

            next_state = game_state.apply_action(a)
            if player == 0:
                util[a] = -self.cfr(next_state, p0 * strategy[a], p1, timeout_event, result, iteration)
            else:
                util[a] = -self.cfr(next_state, p0, p1 * strategy[a], timeout_event, result, iteration)
            node_util += strategy[a] * util[a]

        if player == 0:
            for a in node.actions:
                node.regret_sum[a] += p1 * (util[a] - node_util)
        else:
            for a in node.actions:
                node.regret_sum[a] += p0 * (util[a] - node_util)

        logger.debug(f"cfr returning for info_set: {info_set}, node_util: {node_util}")
        return node_util

    def train(self, timeout_event: Event, result: Dict) -> None:
        for i in range(self.iterations):
            if timeout_event.is_set():
                logger.info(f"Training interrupted after {i} iterations due to timeout.")
                break

            all_cards = Card.get_all_cards()
            random.shuffle(all_cards)
            game_state = GameState(deck=all_cards)
            game_state.selected_cards = Hand(all_cards[:5])
            self.cfr(game_state, 1, 1, timeout_event, result, i + 1)

            if (i + 1) % self.save_interval == 0:
                logger.info(f"Iteration {i+1} of {self.iterations} complete. Saving progress...")
                self.save_progress()
                if self.check_convergence():
                    logger.info(f"CFR agent converged after {i + 1} iterations.")
                    break

    def check_convergence(self) -> bool:
        for node in self.nodes.values():
            avg_strategy = node.get_average_strategy()
            for action, prob in avg_strategy.items():
                if abs(prob - 1.0 / len(node.actions)) > self.stop_threshold:
                    return False
        return True

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        logger.debug("Inside get_move")
        actions = game_state.get_actions()
        logger.debug(f"Available actions: {actions}")

        if not actions:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available, returning error.")
            return

        info_set = game_state.get_information_set()
        logger.debug(f"Info set: {info_set}")

        if info_set in self.nodes:
            strategy = self.nodes[info_set].get_average_strategy()
            logger.debug(f"Strategy: {strategy}")
            best_move = max(strategy, key=strategy.get) if strategy else None
        else:
            logger.debug("Info set not found in nodes, choosing random action.")
            best_move = random.choice(actions) if actions else None

        logger.debug(f"Selected move: {best_move}")
        result["move"] = best_move

    def evaluate_move(self, game_state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float:
        """Оценивает ход, используя комбинацию обученной стратегии MCCFR и эвристики."""
        next_state = game_state.apply_action(action)
        info_set = next_state.get_information_set()

        if info_set in self.nodes:
            node = self.nodes[info_set]
            strategy = node.get_average_strategy()
            expected_value = 0
            for a, prob in strategy.items():
                if timeout_event.is_set():
                    return 0
                expected_value += prob * self.get_action_value(next_state, a, timeout_event)
            return expected_value
        else:
            return self.shallow_search(next_state, 2, timeout_event)

    def shallow_search(self, state: GameState, depth: int, timeout_event: Event) -> float:
        """Поверхностный поиск с ограниченной глубиной."""
        if depth == 0 or state.is_terminal() or timeout_event.is_set():
            return self.baseline_evaluation(state)

        best_value = float("-inf")
        for action in state.get_actions():
            if timeout_event.is_set():
                return 0
            value = -self.shallow_search(state.apply_action(action), depth - 1, timeout_event)
            best_value = max(best_value, value)
        return best_value

    def get_action_value(self, state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> float:
        """Оценивает ценность действия, усредняя результаты нескольких симуляций."""
        num_simulations = 10
        total_score = 0

        for _ in range(num_simulations):
            if timeout_event.is_set():
                return 0
            simulated_state = state.apply_action(action)
            while not simulated_state.is_terminal():
                actions = simulated_state.get_actions()
                if not actions:
                    break
                random_action = random.choice(actions)
                simulated_state = simulated_state.apply_action(random_action)
            total_score += self.baseline_evaluation(simulated_state)

        return total_score / num_simulations if num_simulations > 0 else 0

    def calculate_potential(self, cards: List[Card], line: str, board: Board, available_cards: List[Card]) -> float:
        """Calculates the potential for improvement of a given hand."""
        potential = 0
        num_cards = len(cards)

        if num_cards < 5 and line != "top":
            if self.is_straight_potential(cards, available_cards):
                potential += 0.5
            if self.is_flush_potential(cards, available_cards):
                potential += 0.7

        if num_cards == 2 and line == "top":
            if self.is_pair_potential(cards, available_cards):
                potential += 0.3

        return potential

    def is_flush_potential(self, cards: List[Card], available_cards: List[Card]) -> bool:
        """Checks if there's potential to make a flush."""
        if len(cards) < 2:
            return False

        suit_counts = defaultdict(int)
        for card in cards:
            suit_counts[card.suit] += 1

        for suit, count in suit_counts.items():
            if count >= 2:
                remaining_needed = 5 - count
                available_of_suit = sum(1 for card in available_cards if card.suit == suit)
                if available_of_suit >= remaining_needed:
                    return True
        return False

    def is_straight_potential(self, cards: List[Card], available_cards: List[Card]) -> bool:
        """Checks if there's potential to make a straight."""
        if len(cards) < 2:
            return False

        ranks = sorted([Card.RANKS.index(card.rank) for card in cards])
        for i in range(len(ranks) - 1):
            if ranks[i + 1] - ranks[i] == 1:
                return True

        if len(ranks) >= 2:
            for i in range(len(ranks) - 1):
                if ranks[i + 1] - ranks[i] == 2:
                    needed_rank = ranks[i] + 1
                    if any(Card.RANKS.index(card.rank) == needed_rank for card in available_cards):
                        return True

        if len(ranks) >= 2:
            for i in range(len(ranks) - 1):
                if ranks[i + 1] - ranks[i] == 3:
                    needed_ranks = [ranks[i] + 1, ranks[i] + 2]
                    if sum(1 for card in available_cards if Card.RANKS.index(card.rank) in needed_ranks) >= 1:
                        return True

        if ranks == [0, 1, 2, 3]:
            if any(card.rank == "A" for card in available_cards):
                return True
        if ranks == [0, 1, 2, 12]:
            if any(card.rank == "4" for card in available_cards):
                return True
        if ranks == [0, 1, 11, 12]:
            if any(card.rank == "3" for card in available_cards):
                return True
        if ranks == [0, 10, 11, 12]:
            if any(card.rank == "T" for card in available_cards):
                return True

        return False

    def is_pair_potential(self, cards: List[Card], available_cards: List[Card]) -> bool:
        """Checks if there's potential to make a set (three of a kind) from a pair."""
        if len(cards) != 2:
            return False

        if cards[0].rank == cards[1].rank:
            rank = cards[0].rank
            if sum(1 for card in available_cards if card.rank == rank) >= 1:
                return True

        return False

    def evaluate_line_strength(self, cards: List[Card], line: str) -> float:
        """Evaluates the strength of a line with more granularity."""
        if not cards:
            return 0

        rank, _ = self.evaluate_hand(cards)
        score = 0

        if line == "top":
            if rank == 7:  # Three of a Kind
                score = 15 + Card.RANKS.index(cards[0].rank) * 0.1
            elif rank == 8:  # One Pair
                score = 5 + self.get_pair_bonus(cards)
            elif rank == 9:  # High Card
                score = 1 + self.get_high_card_bonus(cards)
        elif line == "middle":
            if rank == 1:  # Royal Flush
                score = 150
            elif rank == 2:  # Straight Flush
                score = 100 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 3:  # Four of a Kind
                score = 80 + Card.RANKS.index(cards[1].rank) * 0.1
            elif rank == 4:  # Full House
                score = 60 + Card.RANKS.index(cards[2].rank) * 0.1
            elif rank == 5:  # Flush
                score = 40 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 6:  # Straight
                score = 20 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 7:  # Three of a Kind
                score = 10 + Card.RANKS.index(cards[0].rank) * 0.1
            elif rank == 8:  # Two Pair
                score = 5 + Card.RANKS.index(cards[1].rank) * 0.01 + Card.RANKS.index(cards[3].rank) * 0.001
            elif rank == 9:  # One Pair
                score = 2 + Card.RANKS.index(cards[1].rank) * 0.01
            elif rank == 10:  # High Card
                score = Card.RANKS.index(cards[-1].rank) * 0.001
        elif line == "bottom":
            if rank == 1:  # Royal Flush
                score = 120
            elif rank == 2:  # Straight Flush
                score = 80 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 3:  # Four of a Kind
                score = 60 + Card.RANKS.index(cards[1].rank) * 0.1
            elif rank == 4:  # Full House
                score = 40 + Card.RANKS.index(cards[2].rank) * 0.1
            elif rank == 5:  # Flush
                score = 30 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 6:  # Straight
                score = 15 + Card.RANKS.index(cards[-1].rank) * 0.1
            elif rank == 7:  # Three of a Kind
                score = 8 + Card.RANKS.index(cards[0].rank) * 0.1
            elif rank == 8:  # Two Pair
                score = 4 + Card.RANKS.index(cards[1].rank) * 0.01 + Card.RANKS.index(cards[3].rank) * 0.001
            elif rank == 9:  # One Pair
                score = 1 + Card.RANKS.index(cards[1].rank) * 0.01
            elif rank == 10:  # High Card
                score = Card.RANKS.index(cards[-1].rank) * 0.001

        return score

    def baseline_evaluation(self, state: GameState) -> float:
        """Улучшенная эвристическая оценка состояния игры."""
        if state.is_dead_hand():
            return -1000

        COMBINATION_WEIGHTS = {
            "royal_flush": 100,
            "straight_flush": 90,
            "four_of_a_kind": 80,
            "full_house": 70,
            "flush": 60,
            "straight": 50,
            "three_of_a_kind": 40,
            "two_pair": 30,
            "pair": 20,
            "high_card": 10,
        }

        ROW_MULTIPLIERS = {
            "top": 1.0,
            "middle": 1.2,
            "bottom": 1.5,
        }

        total_score = 0

        def evaluate_partial_combination(cards: List[Card], row_type: str) -> float:
            """Оценка потенциала неполной комбинации"""
            if not cards:
                return 0

            score = 0
            ranks = [card.rank for card in cards]
            suits = [card.suit for card in cards]

            suit_counts = Counter(suits)
            max_suit_count = max(suit_counts.values()) if suit_counts else 0
            if row_type in ["middle", "bottom"]:
                if max_suit_count >= 3:
                    score += 15 * max_suit_count

            rank_values = sorted([Card.RANKS.index(rank) for rank in ranks])
            if len(rank_values) >= 3:
                gaps = sum(rank_values[i + 1] - rank_values[i] - 1 for i in range(len(rank_values) - 1))
                if gaps <= 2:
                    score += 10 * (5 - gaps)

            rank_counts = Counter(ranks)
            for rank, count in rank_counts.items():
                rank_value = Card.RANKS.index(rank)
                if count == 2:
                    score += 20 + rank_value
                elif count == 3:
                    score += 40 + rank_value * 1.5

            return score

        rows = {"top": state.board.top, "middle": state.board.middle, "bottom": state.board.bottom}
        for row_name, cards in rows.items():
            row_score = 0

            combination = self.identify_combination(cards)
            if combination:
                row_score += COMBINATION_WEIGHTS[combination]

            potential_score = evaluate_partial_combination(cards, row_name)
            row_score += potential_score

            max_cards = {"top": 3, "middle": 5, "bottom": 5}
            if len(cards) > max_cards[row_name]:
                row_score -= 50

            row_score *= ROW_MULTIPLIERS[row_name]

            total_score += row_score

        if self.is_bottom_stronger_than_middle(state):
            total_score += 30
        if self.is_middle_stronger_than_top(state):
            total_score += 20

        if not self.check_row_strength_rule(state):
            total_score -= 100

        for card in state.discarded_cards:
            rank_value = Card.RANKS.index(card.rank)
            total_score -= rank_value * 0.5

        return total_score

    def identify_combination(self, cards: List[Card]) -> Optional[str]:
        """Определяет тип комбинации."""
        if not cards:
            return None
        if len(cards) < 3 and len(cards) != 5:
            return None
        if len(cards) == 3:
            if self.is_three_of_a_kind(cards):
                return "three_of_a_kind"
            if self.is_one_pair(cards):
                return "pair"
            else:
                return "high_card"
        if self.is_royal_flush(cards):
            return "royal_flush"
        elif self.is_straight_flush(cards):
            return "straight_flush"
        elif self.is_four_of_a_kind(cards):
            return "four_of_a_kind"
        elif self.is_full_house(cards):
            return "full_house"
        elif self.is_flush(cards):
            return "flush"
        elif self.is_straight(cards):
            return "straight"
        elif self.is_three_of_a_kind(cards):
            return "three_of_a_kind"
        elif self.is_two_pair(cards):
            return "two_pair"
        elif self.is_one_pair(cards):
            return "pair"
        else:
            return "high_card"

    def is_bottom_stronger_than_middle(self, state: GameState) -> bool:
        """Проверяет, сильнее ли нижний ряд среднего."""
        if len(state.board.bottom) < 5 or len(state.board.middle) < 5:
            return False
        bottom_rank, _ = self.evaluate_hand(state.board.bottom)
        middle_rank, _ = self.evaluate_hand(state.board.middle)
        return bottom_rank <= middle_rank

    def is_middle_stronger_than_top(self, state: GameState) -> bool:
        """Проверяет, сильнее ли средний ряд верхнего."""
        if len(state.board.middle) < 5 or len(state.board.top) < 3:
            return False
        middle_rank, _ = self.evaluate_hand(state.board.middle)
        top_rank, _ = self.evaluate_hand(state.board.top)
        return middle_rank <= top_rank

    def check_row_strength_rule(self, state: GameState) -> bool:
        """Проверяет, соблюдается ли правило силы рядов (bottom >= middle >= top)."""
        if not state.board.is_full():
            return True

        top_rank, _ = self.evaluate_hand(state.board.top)
        middle_rank, _ = self.evaluate_hand(state.board.middle)
        bottom_rank, _ = self.evaluate_hand(state.board.bottom)

        return bottom_rank <= middle_rank <= top_rank

    def save_progress(self) -> None:
        data = {
            "nodes": self.nodes,
            "iterations": self.iterations,
            "stop_threshold": self.stop_threshold,
        }
        utils.save_ai_progress(data, "cfr_data.pkl")

    def load_progress(self) -> None:
        data = utils.load_ai_progress("cfr_data.pkl")
        if data:
            self.nodes = data["nodes"]
            self.iterations = data["iterations"]
            self.stop_threshold = data.get("stop_threshold", 0.0001)


class RandomAgent:
    def __init__(self):
        pass

    def get_move(self, game_state: GameState, timeout_event: Event, result: Dict) -> None:
        """Chooses a random valid move."""
        logger.debug("Inside RandomAgent get_move")
        actions = game_state.get_actions()
        logger.debug(f"Available actions: {actions}")

        if not actions:
            result["move"] = {"error": "Нет доступных ходов"}
            logger.debug("No actions available, returning error.")
            return

        best_move = random.choice(actions) if actions else None
        logger.debug(f"Selected move: {best_move}")
        result["move"] = best_move

    def evaluate_move(self, game_state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> None:
        pass

    def shallow_search(self, state: GameState, depth: int, timeout_event: Event) -> None:
        pass

    def get_action_value(self, state: GameState, action: Dict[str, List[Card]], timeout_event: Event) -> None:
        pass

    def calculate_potential(
        self, cards: List[Card], line: str, board: Board, available_cards: List[Card]
    ) -> None:
        pass

    def is_flush_potential(self, cards: List[Card], available_cards: List[Card]) -> None:
        pass

    def is_straight_potential(self, cards: List[Card], available_cards: List[Card]) -> None:
        pass

    def is_pair_potential(self, cards: List[Card], available_cards: List[Card]) -> None:
        pass

    def evaluate_line_strength(self, cards: List[Card], line: str) -> None:
        pass

    def baseline_evaluation(self, state: GameState) -> None:
        pass

    def identify_combination(self, cards: List[Card]) -> None:
        pass

    def is_bottom_stronger_than_middle(self, state: GameState) -> None:
        pass

    def is_middle_stronger_than_top(self, state: GameState) -> None:
        pass

    def check_row_strength_rule(self, state: GameState) -> None:
        pass

    def save_progress(self) -> None:
        pass

    def load_progress(self) -> None:
        pass
