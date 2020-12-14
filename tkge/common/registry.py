"""
`tkge.common.registry.Registrable`
"""
from collections import defaultdict
from typing import Type, TypeVar, Callable, Dict, Tuple, Optional, List

from tkge.common.configurable import Configurable
from tkge.common.error import ConfigurationError

import logging
import importlib
logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Registrable")


class Registrable(Configurable):
    """
    Any class that inherits from :class:`Registrable` gains access to a named registry for its subclasses.
    """

    _registry: Dict[Type, Dict[str, Tuple[Type, Optional[str]]]] = defaultdict(dict)
    default_implementation: Optional[str] = None

    @classmethod
    def register(cls: Type[T], name: str, constructor: str = None, overwrite: bool = False):
        """
        Register a subclass

        name: str
            Name to register
        constructor: str
            callback function
        overwrite: bool
            If true, override the existing class under the same name.
            Else, throw an error.
        """

        registry = Registrable._registry[cls]

        def register_subclass(subclass: Type[T]):
            if name in registry:
                if overwrite:
                    msg = (
                        f"{name} has already been registered as {registry[name][0].__name__}, but "
                        f"exist_ok=True, so overwriting with {cls.__name__}"
                    )
                    logger.info(msg)
                else:
                    msg = (
                        f"Cannot register {name} as {cls.__name__}; "
                        f"name already in use for {registry[name][0].__name__}"
                    )
                    raise ConfigurationError(msg)

            registry[name] = (subclass, constructor)
            return subclass

        return register_subclass

    @classmethod
    def by_name(cls: Type[T], name: str) -> Callable[..., T]:
        """
        Returns a callable function that constructs an argument of the registered class.  Because
        you can register particular functions as constructors for specific names, this isn't
        necessarily the `__init__` method of some class.
        """
        logger.debug(f"instantiating registered subclass {name} of {cls}")
        subclass, constructor = cls.resolve_class_name(name)
        if not constructor:
            return subclass
        else:
            return getattr(subclass, constructor)

    @classmethod
    def resolve_class_name(cls: Type[T], name: str) -> Tuple[Type[T], Optional[str]]:
        """
        Returns the subclass that corresponds to the given `name`, along with the name of the
        method that was registered as a constructor for that `name`, if any.

        This method also allows `name` to be a fully-specified module name, instead of a name that
        was already added to the `Registry`.  In that case, you cannot use a separate function as
        a constructor (as you need to call `cls.register()` in order to tell us what separate
        function to use).
        """
        if name in Registrable._registry[cls]:
            subclass, constructor = Registrable._registry[cls][name]
            return subclass, constructor
        elif "." in name:
            # This might be a fully qualified class name, so we'll try importing its "module"
            # and finding it there.
            parts = name.split(".")
            submodule = ".".join(parts[:-1])
            class_name = parts[-1]

            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError:
                raise ConfigurationError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to import module {submodule}"
                )

            try:
                subclass = getattr(module, class_name)
                constructor = None
                return subclass, constructor
            except AttributeError:
                raise ConfigurationError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to find class {class_name} in {submodule}"
                )

        else:
            # is not a qualified class name
            raise ConfigurationError(
                f"{name} is not a registered name for {cls.__name__}. "
                "You probably need to use the --include-package flag "
                "to load your custom code. Alternatively, you can specify your choices "
                """using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} """
                "in which case they will be automatically imported correctly."
            )

    @classmethod
    def list_available(cls) -> List[str]:
        """List default first if it exists"""
        keys = list(Registrable._registry[cls].keys())
        default = cls.default_implementation

        if default is None:
            return keys
        elif default not in keys:
            raise ConfigurationError(f"Default implementation {default} is not registered")
        else:
            return [default] + [k for k in keys if k != default]

