"""
Deprecation shims for the names renamed to PEP 8 snake_case in Lightweaver 1.0.

The old names (see `lightweaver._renames.RENAMES`) keep working, emitting a
`LightweaverDeprecationWarning` that names the replacement, and will be removed
in a future release. Scripts can be updated automatically with
``python -m lightweaver.migrate <files>``, and the warnings can be silenced with
`silence_deprecations`.
"""

import dataclasses
import functools
import inspect
import warnings
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Set, TypeVar

from ._renames import RENAMES

__all__ = [
    'LightweaverDeprecationWarning',
    'accepts_old_kwargs',
    'deprecated_alias',
    'deprecated_names',
    'remap_old_keys',
    'silence_deprecations',
]

F = TypeVar('F', bound=Callable)
T = TypeVar('T', bound=type)

# new name -> old names (several old names can map to one new name, e.g. max_iter).
_OLD_NAMES: Dict[str, List[str]] = defaultdict(list)
for _old, _new in RENAMES.items():
    _OLD_NAMES[_new].append(_old)

# Callable short name (function, method, or class for its constructor) -> old
# keyword names it accepts. Used by `lightweaver.migrate`.
KWARG_REGISTRY: Dict[str, Set[str]] = defaultdict(set)


class LightweaverDeprecationWarning(FutureWarning):
    """
    Warning for deprecated Lightweaver names. A `FutureWarning` subclass, so it
    is shown by default.
    """

    pass


def _warn(old: str, new: str, stacklevel: int):
    warnings.warn(
        f'`{old}` is deprecated, use `{new}` instead (it will be removed in a future '
        'version of Lightweaver). Run `python -m lightweaver.migrate <files>` to update '
        'scripts automatically.',
        LightweaverDeprecationWarning,
        stacklevel=stacklevel + 1,
    )


def silence_deprecations():
    """
    Silence the warnings for deprecated Lightweaver names.
    """
    warnings.filterwarnings('ignore', category=LightweaverDeprecationWarning)


def remap_old_keys(d: dict, stacklevel: int = 2, renames: Dict[str, str] = RENAMES) -> dict:
    """
    Return a copy of a keyword dict (e.g. the `kwargs` of a `Context` state
    dict) with deprecated keys renamed (with a warning). If both an old key and
    its replacement are present, the old key's value wins: it can only have been
    added by the caller to override the stored value (e.g.
    ``dict(sd['kwargs'], formalSolver=...)``).

    `renames` is the old -> new table to apply (default: `RENAMES`).
    """
    result = dict(d)
    for old in [k for k in result if k in renames]:
        new = renames[old]
        _warn(old, new, stacklevel=stacklevel + 1)
        result[new] = result.pop(old)
    return result


def _param_names(func: Callable) -> Set[str]:
    try:
        return set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        code = func.__code__
        return set(code.co_varnames[: code.co_argcount + code.co_kwonlyargcount])


def accepts_old_kwargs(func: F = None, *, name: str = None) -> F:
    """
    Decorator remapping deprecated keyword arguments to their new names (with a
    warning). Passing both the old and new name raises a `TypeError`.

    Parameters
    ----------
    func : callable
        The function to wrap.
    name : str, optional
        The name to register for `lightweaver.migrate` (default: the
        function's `__name__`, or the class name for `__init__`).
    """
    if func is None:
        return functools.partial(accepts_old_kwargs, name=name)

    params = _param_names(func)
    mapping = {
        old: new
        for new in params
        if new in _OLD_NAMES
        for old in _OLD_NAMES[new]
        if old not in params
    }
    if not mapping:
        return func

    if name is None:
        name = func.__name__
        if name == '__init__':
            name = func.__qualname__.split('.')[-2]
    KWARG_REGISTRY[name].update(mapping)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for old in mapping.keys() & kwargs.keys():
            new = mapping[old]
            if new in kwargs:
                raise TypeError(
                    f'{func.__qualname__}() got both `{old}` and its replacement `{new}`'
                )
            _warn(old, new, stacklevel=2)
            kwargs[new] = kwargs.pop(old)
        return func(*args, **kwargs)

    wrapper.__lw_renamed_kwargs__ = mapping
    return wrapper


class deprecated_alias:  # noqa: N801 (descriptor, lowercase like `property`)
    """
    Descriptor forwarding a deprecated attribute name to its replacement (with a
    warning) for reading, writing and deleting.
    """

    def __init__(self, new: str):
        self.new = new
        self.old = None

    def __set_name__(self, owner, name):
        self.old = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        _warn(self.old, self.new, stacklevel=2)
        return getattr(obj, self.new)

    def __set__(self, obj, value):
        _warn(self.old, self.new, stacklevel=2)
        setattr(obj, self.new, value)

    def __delete__(self, obj):
        _warn(self.old, self.new, stacklevel=2)
        delattr(obj, self.new)


def deprecated_names(cls: T = None, *, attrs: Iterable[str] = ()) -> T:
    """
    Class decorator providing the deprecated names for a class: every method
    (including `__init__`) with renamed parameters accepts the old keyword
    names, and old-name aliases are added for renamed dataclass fields,
    properties, methods, and the instance attributes listed in `attrs`.

    For dataclasses, apply it above `@dataclass`. It must be applied to each
    class in a hierarchy that defines (or, for dataclasses, generates) methods.

    Parameters
    ----------
    cls : type
        The class to decorate.
    attrs : iterable of str, optional
        New names of plain instance attributes (set in `__init__`) to alias.
    """
    if cls is None:
        return functools.partial(deprecated_names, attrs=attrs)

    for key, value in list(vars(cls).items()):
        if isinstance(value, (staticmethod, classmethod)):
            wrapped = accepts_old_kwargs(
                value.__func__, name=cls.__name__ if key == '__init__' else key
            )
            if wrapped is not value.__func__:
                setattr(cls, key, type(value)(wrapped))
        elif inspect.isfunction(value):
            wrapped = accepts_old_kwargs(value, name=cls.__name__ if key == '__init__' else key)
            if wrapped is not value:
                setattr(cls, key, wrapped)

    names = set(attrs)
    if dataclasses.is_dataclass(cls):
        names.update(f.name for f in dataclasses.fields(cls))
    names.update(
        k for k, v in vars(cls).items() if isinstance(v, property) or inspect.isfunction(v)
    )
    for new in names:
        for old in _OLD_NAMES.get(new, ()):
            if old not in vars(cls):
                alias = deprecated_alias(new)
                setattr(cls, old, alias)
                alias.__set_name__(cls, old)
    return cls
