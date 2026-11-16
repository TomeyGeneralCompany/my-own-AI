"""Standard plug-ins; registration is not activation."""
from .finance_module import Finance
from .business_module import Business

LABELS = {"conversation":"Conversation","writing":"Author","coding":"Engineer",
          "research":"Researcher","finance":"Stock Trader","business":"Business Manager"}


def standard_plugins(workspace):
    return [Finance(workspace), Business(workspace)]
