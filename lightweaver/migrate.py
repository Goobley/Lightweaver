"""
Update scripts to the snake_case names introduced in Lightweaver 1.0.

Usage::

    python -m lightweaver.migrate [--inplace] PATH [PATH ...]

PATHs can be files or directories (searched recursively for ``*.py``). By
default a unified diff of the proposed changes is printed and nothing is
modified; ``--inplace`` modifies the files in place.

What is renamed:

- keyword arguments in calls to Lightweaver functions, methods and classes
  that accept the old names (e.g. ``lw.Context(..., conserveCharge=True)``);
- attribute accesses using an old name (e.g. ``ctx.activeAtoms``,
  ``atmos.nHTot``), in both reads and writes;
- keys of a `Context` state dict's ``kwargs`` (``sd['kwargs']['eqPops']`` and
  ``dict(sd['kwargs'], eqPops=...)``);
- keys of a ``time_dependent_data`` dict literal passed to ``nr_post_update``
  (``{'dt': dt, 'nPrev': prev_state}``).

Bare names (local variables such as ``eqPops = ...``) and calls to your own
functions are never touched. Attribute renames apply to any object, so if your
own classes reuse a Lightweaver attribute name it will be renamed consistently;
these are listed in the summary for review. Formatting and comments are
preserved.
"""

import argparse
import ast
import difflib
import importlib
import pkgutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

from ._renames import NR_DICT_KEY_RENAMES, RENAMES

Edit = Tuple[int, int, str, str]  # (line index, byte column, old, new)


def build_kwarg_registry() -> Dict[str, Set[str]]:
    """
    Import Lightweaver's modules so every shimmed callable registers the old
    keyword names it accepts, and return the registry (callable short name ->
    old keyword names).
    """
    import lightweaver

    from .deprecation import KWARG_REGISTRY

    for mod in pkgutil.iter_modules(lightweaver.__path__):
        # libenkiTS is a plain shared library; on Windows it carries a stub
        # PyInit that returns NULL, so importing it raises SystemError.
        if mod.name in ('migrate', 'libenkiTS'):
            continue
        try:
            importlib.import_module(f'lightweaver.{mod.name}')
        except ImportError:
            # Optional dependencies (e.g. crtaf).
            pass
    return {k: set(v) for k, v in KWARG_REGISTRY.items()}


def _is_kwargs_subscript(node: ast.expr) -> bool:
    """Whether `node` is `<expr>['kwargs']`, i.e. the kwargs of a Context state dict."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == 'kwargs'
    )


def _callee_name(func: ast.expr):
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def find_edits(source: str, registry: Dict[str, Set[str]]) -> Tuple[List[Edit], List[Edit]]:
    """
    Return the (keyword, attribute) edits for a source file.
    """
    tree = ast.parse(source)
    kw_edits: List[Edit] = []
    attr_edits: List[Edit] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            accepted = registry.get(_callee_name(node.func), ())
            if _callee_name(node.func) == 'dict' and any(
                _is_kwargs_subscript(a) for a in node.args
            ):
                # dict(sd['kwargs'], oldName=...) on a Context state dict.
                accepted = RENAMES
            for kw in node.keywords:
                if kw.arg in accepted:
                    kw_edits.append((kw.lineno - 1, kw.col_offset, kw.arg, RENAMES[kw.arg]))
                if (
                    _callee_name(node.func) == 'nr_post_update'
                    and kw.arg in ('time_dependent_data', 'timeDependentData')
                    and isinstance(kw.value, ast.Dict)
                ):
                    # ctx.nr_post_update(time_dependent_data={'nPrev': ...}): rename inside the quotes.
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant) and key.value in NR_DICT_KEY_RENAMES:
                            new = NR_DICT_KEY_RENAMES[key.value]
                            kw_edits.append((key.lineno - 1, key.col_offset + 1, key.value, new))
        elif (
            isinstance(node, ast.Subscript)
            and _is_kwargs_subscript(node.value)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in RENAMES
        ):
            # sd['kwargs']['oldName'] on a Context state dict: rename inside the quotes.
            key = node.slice
            kw_edits.append((key.lineno - 1, key.col_offset + 1, key.value, RENAMES[key.value]))
        elif isinstance(node, ast.Attribute) and node.attr in RENAMES:
            col = node.end_col_offset - len(node.attr)
            attr_edits.append((node.end_lineno - 1, col, node.attr, RENAMES[node.attr]))
    return kw_edits, attr_edits


def apply_edits(source: str, edits: List[Edit]) -> str:
    # ast offsets are UTF-8 byte offsets, so edit the encoded lines.
    lines = source.encode('utf-8').splitlines(keepends=True)
    for line, col, old, new in sorted(edits, reverse=True):
        current = lines[line][col : col + len(old)]
        if current != old.encode('utf-8'):
            raise ValueError(f'Unexpected source at line {line + 1}: {current!r} != {old!r}')
        lines[line] = lines[line][:col] + new.encode('utf-8') + lines[line][col + len(old) :]
    return b''.join(lines).decode('utf-8')


def migrate_source(
    source: str, registry: Dict[str, Set[str]]
) -> Tuple[str, List[Edit], List[Edit]]:
    """
    Return the migrated source and the (keyword, attribute) edits made.
    """
    kw_edits, attr_edits = find_edits(source, registry)
    return apply_edits(source, kw_edits + attr_edits), kw_edits, attr_edits


def _python_files(paths: List[str]):
    for p in paths:
        path = Path(p)
        if path.is_dir():
            yield from sorted(
                f for f in path.rglob('*.py') if not any(part.startswith('.') for part in f.parts)
            )
        else:
            yield path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m lightweaver.migrate',
        description='Update scripts to the snake_case names introduced in Lightweaver 1.0.',
    )
    parser.add_argument('paths', nargs='+', help='Files or directories to migrate.')
    parser.add_argument(
        '--inplace', action='store_true', help='Modify the files in place (default: print a diff).'
    )
    args = parser.parse_args(argv)

    registry = build_kwarg_registry()
    names: Counter = Counter()
    attr_names: Counter = Counter()
    changed = 0
    errors = 0
    for path in _python_files(args.paths):
        try:
            source = path.read_text(encoding='utf-8')
            new, kw_edits, attr_edits = migrate_source(source, registry)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            print(f'{path}: skipped ({e})', file=sys.stderr)
            errors += 1
            continue
        if new == source:
            continue
        changed += 1
        names.update((e[2], e[3]) for e in kw_edits)
        attr_names.update((e[2], e[3]) for e in attr_edits)
        if args.inplace:
            path.write_text(new, encoding='utf-8')
            print(f'{path}: {len(kw_edits)} keyword and {len(attr_edits)} attribute renames')
        else:
            sys.stdout.writelines(
                difflib.unified_diff(
                    source.splitlines(keepends=True),
                    new.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )

    print(
        f'\n{changed} file(s) {"updated" if args.inplace else "would be updated"}.', file=sys.stderr
    )
    if names:
        print(
            'Keyword arguments and dict keys: '
            + ', '.join(f'{old} -> {new} ({v})' for (old, new), v in names.most_common()),
            file=sys.stderr,
        )
    if attr_names:
        print(
            'Attributes (also applied to your own objects that use these names; please review): '
            + ', '.join(f'{old} -> {new} ({v})' for (old, new), v in attr_names.most_common()),
            file=sys.stderr,
        )
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
