import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "predefined"


@dataclass
class Process:
    spec: dict
    implementation: Optional[Callable] = None
    namespace: str = DEFAULT_NAMESPACE


class ProcessRegistry(MutableMapping):
    """
    The process registry is a dictionary mapping from namespace to a dictionary of process_id to Process.
    It allows registering aliases for process_ids.
    """

    def __init__(self, wrap_funcs: Optional[list] = None, *args, **kwargs):
        """wrap_funcs: list of decorators to apply to all registered processes."""
        self.store = dict()  # type: dict[str, dict[str, Process]]
        self.aliases = dict()  # type: dict[str, dict[str, str]]

        if wrap_funcs is None:
            wrap_funcs = []
        self.wrap_funcs = wrap_funcs

        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        t_namespace, t_key = self._keytransform(key)
        if t_namespace in self.store:
            if t_key is None:  # If no key is provided, return the whole namespace
                return self.store[t_namespace]
            else:  # If a key is provided, return the specific process
                if (
                    t_namespace in self.aliases
                    and (t_namespace, t_key) in self.aliases[t_namespace]
                ):  # Check if provided key has an alias
                    _, t_key = self.aliases[t_namespace][(t_namespace, t_key)]
                if t_key in self.store[t_namespace]:
                    return self.store[t_namespace][t_key]
                else:
                    raise KeyError(
                        f"Process {t_key} not found in namespace {t_namespace}!"
                    )
        else:
            raise KeyError(f"Namespace {t_namespace} not found in process registry!")

    def __setitem__(self, key, process: Process):
        t_namespace, t_key = self._keytransform(key)

        if t_key is None:  # If no key is provided, set the entire namespace
            if type(process) is not dict:
                raise ValueError(
                    f"Expected a dictionary of processes, got {type(process)}"
                )
            else:
                for k, v in process.items():
                    self.__setitem__((t_namespace, k), v)
        else:  # If a key is provided, set the specific process
            if process.implementation is not None:
                decorated_callable = process.implementation
                for wrap_f in self.wrap_funcs:
                    decorated_callable = wrap_f(decorated_callable)
                process.implementation = decorated_callable
            if t_namespace not in self.store:
                self.store[t_namespace] = {}
            self.store[t_namespace][t_key] = process

    def __delitem__(self, key):
        t_namespace, t_key = self._keytransform(key)

        if t_key is None:  # If no key is provided, delete the entire namespace
            del self.store[t_namespace]
        else:  # If a key is provided, delete the specific process
            del self.store[t_namespace][t_key]

    def __iter__(self):
        for namespace_key, namespace_items in self.store.items():
            for item_key, _ in namespace_items.items():
                yield (namespace_key, item_key)

    def __len__(self):
        total_len = sum(len(dct) for dct in self.store.values())
        return total_len

    def _keytransform(self, key):
        """Process the key into a namespace and a process key."""
        if isinstance(key, tuple):
            return str(key[0]).strip("_"), (
                None if key[1] is None else str(key[1]).strip("_")
            )
        else:
            return DEFAULT_NAMESPACE, None if key is None else str(key).strip("_")

    def add_alias(self, process_id: str, alias: str, namespace: str = DEFAULT_NAMESPACE):
        """
        Method to allow adding aliases to processes.
        This can be useful for not-yet standardised processes, where an OpenEO client might use a different process_id than the backend.
        """
        t_namespace, t_key = self._keytransform((namespace, process_id))
        if t_key not in self.store[t_namespace]:
            raise ValueError(
                f"Could not add alias {alias} -> {process_id}, because process_id {process_id} was not found in the process registry."
            )

        # Add the alias to the self.aliases dict
        self.aliases.setdefault(t_namespace, {})
        self.aliases[t_namespace][self._keytransform(alias)] = t_namespace, t_key
        logger.debug(
            f"Added alias {alias} -> {process_id} to process registry under namespace {t_namespace}."
        )

    def add_wrap_func(self, wrap_func: Callable):
        self.wrap_funcs.append(wrap_func)

        # Wrap all existing processes retroactively
        for _, process in self.items():
            process.implementation = wrap_func(process.implementation)
