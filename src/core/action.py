from enum import Enum, auto


class ActionType(Enum):
    PROPOSE = auto()
    COMMENT = auto()
    SUPPORT = auto()
    OPPOSE = auto()
    VOTE = auto()
    DO_NOTHING = auto()
