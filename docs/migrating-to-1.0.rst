Migrating to Lightweaver 1.0
============================

Lightweaver 1.0 renames its public API to PEP 8 snake_case, e.g.
``lw.Context(atmos, spect, eqPops, conserveCharge=True)`` becomes
``lw.Context(atmos, spect, eq_pops, conserve_charge=True)``, and
``ctx.activeAtoms`` becomes ``ctx.active_atoms``.

**Your existing scripts will continue working for now.** The old names are still accepted, and emit a ``LightweaverDeprecationWarning`` e.g.::

    LightweaverDeprecationWarning: `conserveCharge` is deprecated, use
    `conserve_charge` instead (it will be removed in a future version of
    Lightweaver). Run `python -m lightweaver.migrate <files>` to update scripts
    automatically.

Each warning is shown once per line of your code that uses an old name. The old
names will be removed in the future.

Silencing the warnings
----------------------

To run old scripts without the warnings (to auto-update them, see below), call:

.. code-block:: python

    import lightweaver as lw
    lw.silence_deprecations()

or equivalently, with the standard library:

.. code-block:: python

    import warnings
    from lightweaver import LightweaverDeprecationWarning
    warnings.filterwarnings('ignore', category=LightweaverDeprecationWarning)

Updating scripts automatically
------------------------------

Lightweaver includes a tool that rewrites scripts to the new names::

    python -m lightweaver.migrate my_script.py my_analysis_dir/

By default this only prints a diff of the proposed changes. When you're happy
with them, modify the files in place with ``--inplace``::

    python -m lightweaver.migrate --inplace my_script.py my_analysis_dir/

The tool renames:

- keyword arguments in calls to Lightweaver functions, methods and classes
  (``lw.iterate_ctx_se(ctx, popsTol=1e-3)`` → ``pops_tol=1e-3``), including
  atomic model definitions (``LinearCoreExpWings(qCore=..., qWing=...)``);
- attributes (``ctx.activeAtoms``, ``atmos.nHTot``, ``update.dJMax``);
- keys of a ``Context`` state dict's ``kwargs`` (``sd['kwargs']['eqPops']``);
- the ``'nPrev'`` key of a ``time_dependent_data`` dict passed to
  ``nr_post_update`` (``{'dt': dt, 'nPrev': prev_state}``).

It never renames your own variables or functions: ``eqPops = ...`` and
``def synth(eqPops): ...`` are left as they are. Attribute renames are applied
to any object, so if your own classes reuse a Lightweaver attribute name (e.g.
``self.eqPops``), that attribute is renamed consistently too; such renames are
listed in the summary printed at the end for you to review. Formatting and
comments are preserved.

Naming rule
-----------

- Ordinary camelCase words become snake_case: ``eqPops`` → ``eq_pops``,
  ``vlosMu`` → ``vlos_mu``.
- Physics symbols keep their case: ``JTol`` → ``J_tol``, ``dJMax`` →
  ``dJ_max``, ``phiQ`` → ``phi_Q``, ``gammaB`` → ``gamma_B``.
- Element symbols are lowercase: ``nHTot`` → ``nh_tot``, ``hPops`` →
  ``h_pops``, ``massPerH`` → ``mass_per_h``.
- Names that are only a count or a symbol are unchanged: ``Nspace``,
  ``Nrays``, ``Nthreads``, ``Aji``, ``Gamma``, ``B``, ``T``.
- Counts with a trailing word keep the ``N`` attached to the first word:
  ``NlambdaGen`` → ``Nlambda_gen``.
- ``NmaxIter`` (``iterate_ctx_se``) and ``maxIter`` both become ``max_iter``.

Other changes
-------------

- Subclasses of Lightweaver's extension points (``CollisionalRates``,
  ``LineBroadener``, ``BackgroundProvider``, ``BoundaryCondition``,
  ``ConvergenceCriteria``) that use the old parameter names in their method
  signatures keep working, as Lightweaver calls these methods positionally.
- The keys of ``Context.kwargs`` (and of a state dict's ``kwargs``) use the new
  names. Old keys are still accepted by
  ``Context.construct_from_state_dict_with``.
- The ``time_dependent_data`` dict passed to ``Context.nr_post_update`` now
  uses the key ``'n_prev'`` (previously ``'nPrev'``). The old key is still
  accepted, with a warning.
- Pickled ``Context`` objects and state dicts from earlier versions can't be
  loaded. These have only ever been intended as transient caches; recreate them
  with the new version.

All renamed names
-----------------

.. include:: _generated_renames.rst
