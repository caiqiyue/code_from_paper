import hashlib
import diskcache as dc
from abc import ABC, abstractmethod

class EngineLM(ABC):
    """Abstract interface implemented by all language-model backends."""
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str
    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        """Generate a response with the configured model backend."""
        pass

    def __call__(self, *args, **kwargs):
        """Invoke the backend and return the generated response."""
        pass


class CachedEngine:
    """Mixin that stores prompt-response pairs in a disk-backed cache."""
    def __init__(self, cache_path):
        """Initialize the CachedEngine instance."""
        super().__init__()
        self.cache_path = cache_path
        self.cache = dc.Cache(cache_path)

    def _hash_prompt(self, prompt: str):
        """Compute a stable hash for a prompt string."""
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str):
        """Return a cached response when the prompt has been seen before."""
        if prompt in self.cache:
            return self.cache[prompt]
        else:
            return None

    def _save_cache(self, prompt: str, response: str):
        """Store a prompt-response pair in the cache."""
        self.cache[prompt] = response

    def __getstate__(self):
        # Remove the cache from the state before pickling
        """Prepare the object state for pickling."""
        state = self.__dict__.copy()
        del state['cache']
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling
        """Restore the object state after unpickling."""
        self.__dict__.update(state)
        self.cache = dc.Cache(self.cache_path)
