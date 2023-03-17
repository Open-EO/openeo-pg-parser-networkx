import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Process:
    spec: dict
    implementation: Optional[Callable] = None
    namespace: str = "predefined"


class ProcessRegistry(MutableMapping):
    """
    The process registry is basically a dictionary mapping from process_id to a tuple callable implementation.
    It also allows registering aliases for process_ids.
    """

    def __init__(self, wrap_funcs: Optional[list] = None, *args, **kwargs):
        """wrap_funcs: list of decorators to apply to all registered processes."""
        self.store = dict()  # type: dict[str, Process]
        self.aliases = dict()  # type: dict[str, str]

        if wrap_funcs is None:
            wrap_funcs = []
        self.wrap_funcs = wrap_funcs

        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key) -> Process:
        t_key = self._keytransform(key)
        if t_key in self.store:
            return self.store[t_key]
        if t_key in self.aliases:
            original_key = self.aliases[t_key]
            if original_key in self.store:
                return self.store[original_key]
            else:
                del self.aliases[t_key]

        raise KeyError(f"Key {key} not found in process registry!")

    def __setitem__(self, key, process: Process):
        t_key = self._keytransform(key)

        if process.implementation is not None:
            decorated_callable = process.implementation
            for wrap_f in self.wrap_funcs:
                decorated_callable = wrap_f(decorated_callable)
            process.implementation = decorated_callable
        self.store[t_key] = process

    def __delitem__(self, key):
        t_key = self._keytransform(key)

        del self.store[t_key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        """Some processes are prefixed with an underscore to prevent clashes with built-in names.
        These need to be stripped before being put into the registry."""
        return key.strip("_")

    def add_alias(self, process_id: str, alias: str):
        """
        Method to allow adding aliases to processes.
        This can be useful for not-yet standardised processes, where an OpenEO client might use a different process_id than the backend.
        """

        if process_id not in self.store:
            raise ValueError(
                f"Could not add alias {alias} -> {process_id}, because process_id {process_id} was not found in the process registry."
            )

        # Add the alias to the self.aliases dict
        self.aliases[self._keytransform(alias)] = self._keytransform(process_id)
        logger.debug(f"Added alias {alias} -> {process_id} to process registry.")

    def add_wrap_func(self, wrap_func: Callable):
        self.wrap_funcs.append(wrap_func)

        # Wrap all existing processes retroactively
        for _, process in self.items():
            process.implementation = wrap_func(process.implementation)
