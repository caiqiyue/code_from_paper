import os
import logging
import json
from datetime import datetime
class CustomJsonFormatter(logging.Formatter):
    """JSON log formatter used by the package-level logger."""
    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record into JSON for file-based experiment traces."""
        super(CustomJsonFormatter, self).format(record)
        output = {k: str(v) for k, v in record.__dict__.items()}
        return json.dumps(output, indent=4)

cf = CustomJsonFormatter()
os.makedirs("./logs/", exist_ok=True)
sh = logging.FileHandler(f"./logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
sh.setFormatter(cf)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(sh)

from .variable import Variable
from .loss import TextLoss
from .model import BlackboxLLM
from .engine import EngineLM, get_engine
from .optimizer import TextualGradientDescent, TGD
from .config import set_backward_engine, SingletonBackwardEngine
from .autograd import sum, aggregate

singleton_backward_engine = SingletonBackwardEngine()