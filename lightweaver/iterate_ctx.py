import time
from typing import TYPE_CHECKING, Optional, Type

from .deprecation import accepts_old_kwargs, deprecated_names
from .iteration_update import IterationUpdate

if TYPE_CHECKING:
    from . import Context


@deprecated_names
class ConvergenceCriteria:
    """
    Abstract base class for determining convergence inside `iterate_ctx_se`. A
    derived variant of this class will be instantiated by `iterate_ctx_se`
    dependent upon its arguments. The default implementation is
    `DefaultConvergenceCriteria`.

    Parameters
    ----------
    ctx : Context
        The context being iterated.
    J_tol : float
        The value of J_tol passed to `iterate_ctx_se`.
    pops_tol : float
        The value of pops_tol passed to `iterate_ctx_se`.
    rho_tol : float or None
        The value of rho_tol passed to `iterate_ctx_se`.
    """

    def __init__(self, ctx: 'Context', J_tol: float, pops_tol: float, rho_tol: Optional[float]):
        raise NotImplementedError

    def is_converged(
        self,
        J_update: IterationUpdate,
        pops_update: IterationUpdate,
        prd_update: Optional[IterationUpdate],
    ) -> bool:
        """
        This function takes the IterationUpdate objects from
        `ctx.formal_sol_gamma_matrices` and `ctx.stat_equil` and optionally from
        `ctx.prd_redistribute` (or None).  Should return a bool indicated
        whether the Context is sufficiently converged.
        """
        raise NotImplementedError


@deprecated_names(attrs=('J_tol', 'pops_tol', 'rho_tol'))
class DefaultConvergenceCriteria(ConvergenceCriteria):
    """
    Default ConvergenceCriteria implementation. Usually sufficient for
    statistical equilibrium problems, but you may occasionally need to override
    this.

    Parameters
    ----------
    ctx : Context
        The context being iterated.
    J_tol : float
        The value of J_tol passed to `iterate_ctx_se`.
    pops_tol : float
        The value of pops_tol passed to `iterate_ctx_se`.
    rho_tol : float or None
        The value of rho_tol passed to `iterate_ctx_se`.
    """

    def __init__(self, ctx: 'Context', J_tol: float, pops_tol: float, rho_tol: Optional[float]):
        self.ctx = ctx
        self.J_tol = J_tol
        self.pops_tol = pops_tol
        self.rho_tol = rho_tol

    def is_converged(
        self,
        J_update: IterationUpdate,
        pops_update: IterationUpdate,
        prd_update: Optional[IterationUpdate],
    ) -> bool:
        """
        Returns whether the context is converged.
        """
        updates = [J_update, pops_update]
        if prd_update is not None:
            updates.append(prd_update)

        terminate = True
        for update in updates:
            terminate = terminate and (update.dJ_max < self.J_tol)
            terminate = terminate and (update.dpops_max < self.pops_tol)
            if prd_update and self.rho_tol is not None:
                terminate = terminate and (update.drho_max < self.rho_tol)
        terminate = terminate and self.ctx.crsw_done

        return terminate


@accepts_old_kwargs
def iterate_ctx_se(
    ctx: 'Context',
    Nscatter: int = 3,
    max_iter: int = 2000,
    prd: bool = False,
    J_tol: float = 5e-3,
    pops_tol: float = 1e-3,
    rho_tol: Optional[float] = None,
    prd_iter_tol: float = 1e-2,
    max_prd_sub_iter: int = 3,
    print_interval: float = 0.2,
    quiet: bool = False,
    convergence: Optional[Type[ConvergenceCriteria]] = None,
    return_final_convergence: bool = False,
):
    """
    Iterate a configured Context towards statistical equilibrium solution.

    Parameters
    ----------
    ctx : Context
        The context to iterate.
    Nscatter : int, optional
        The number of lambda iterations to perform for an initial estimate of J
        (default: 3).
    max_iter : int, optional
        The maximum number of iterations (including Nscatter) to take (default:
        2000).
    prd: bool, optional
        Whether to perform PRD subiterations to estimate rho for PRD lines
        (default: False).
    J_tol: float, optional
        The maximum relative change in J from one iteration to the next
        (default: 5e-3).
    pops_tol: float, optional
        The maximum relative change in an atomic population from one iteration
        to the next (default: 1e-3).
    rho_tol: float, optional
        The maximum relative change in rho for a PRD line on the final
        subiteration from one iteration to the next. If None, the change in rho
        will not be considered in judging convergence (default: None).
    prd_iter_tol: float, optional
        The maximum relative change in rho for a PRD line below which PRD
        subiterations will cease for this iteration (default: 1e-2).
    max_prd_sub_iter : int, optional
        The maximum number of PRD subiterations to make, whether or not rho has
        reached the tolerance of prd_iter_tol (which isn't necessary every
        iteration). (Default: 3)
    print_interval : float, optional
        The interval between printing the update size information in seconds. A
        value of 0.0 will print every iteration (default: 0.2).
    quiet : bool, optional
        Overrides any other print arguments and iterates silently if True.
        (Default: False).
    convergence : derived ConvergenceCriteria class, optional
        The ConvergenceCriteria version to be used in determining convergence.
        Will be instantiated by this function, and the `is_converged` method
        will then be used.  (Default: DefaultConvergenceCriteria).
    return_final_convergence : bool, optional
        Whether to return the IterationUpdate objects used in the final
        convergence decision, if True, these will be returned in a list as the
        second return value. (Default: False).

    Returns
    -------
    it : int
        The number of iterations taken.
    finalIterationUpdates : List[IterationUpdate], optional
        The final IterationUpdates computed, if requested by `return_final_convergence`.
    """

    prevPrint = 0.0
    printNow = True
    alwaysPrint = print_interval == 0.0
    startTime = time.time()

    if convergence is None:
        convergence = DefaultConvergenceCriteria
    # NOTE: Call user-overridable hooks positionally, so subclasses written with the pre-1.0
    # parameter names keep working.
    conv = convergence(ctx, J_tol, pops_tol, rho_tol)

    for it in range(max_iter):
        J_update: IterationUpdate = ctx.formal_sol_gamma_matrices()
        if not quiet and (alwaysPrint or ((now := time.time()) >= prevPrint + print_interval)):
            printNow = True
            if not alwaysPrint:
                prevPrint = now

        if not quiet and printNow:
            print(f'-- Iteration {it}:')
            print(J_update.compact_representation())

        if it < Nscatter:
            if not quiet and printNow:
                print('    (Lambda iterating background)')
            # NOTE(cmo): reset print state
            printNow = False
            continue

        pops_update: IterationUpdate = ctx.stat_equil()
        if not quiet and printNow:
            print(pops_update.compact_representation())

        dRhoUpdate: Optional[IterationUpdate]
        if prd:
            dRhoUpdate = ctx.prd_redistribute(max_iter=max_prd_sub_iter, tol=prd_iter_tol)
            if not quiet and printNow and dRhoUpdate is not None:
                print(dRhoUpdate.compact_representation())
        else:
            dRhoUpdate = None

        terminate = conv.is_converged(J_update, pops_update, dRhoUpdate)

        if terminate:
            if not quiet:
                endTime = time.time()
                duration = endTime - startTime
                line = '-' * 80
                if printNow:
                    print('Final Iteration shown above.')
                else:
                    print(line)
                    print(f'Final Iteration: {it}')
                    print(line)
                    print(J_update.compact_representation())
                    print(pops_update.compact_representation())
                    if prd and dRhoUpdate is not None:
                        print(dRhoUpdate.compact_representation())
                print(line)
                print(
                    f'Context converged to statistical equilibrium in {it}'
                    f' iterations after {duration:.2f} s.'
                )
                print(line)
            if return_final_convergence:
                finalConvergence = [J_update, pops_update]
                if prd and dRhoUpdate is not None:
                    finalConvergence.append(dRhoUpdate)
                return it, finalConvergence
            else:
                return it

        # NOTE(cmo): reset print state
        printNow = False
    else:
        if not quiet:
            line = '-' * 80
            endTime = time.time()
            duration = endTime - startTime
            print(line)
            print(f'Final Iteration: {it}')
            print(line)
            print(J_update.compact_representation())
            print(pops_update.compact_representation())
            if prd and dRhoUpdate is not None:
                print(dRhoUpdate.compact_representation())
            print(line)
            print(
                f'Context FAILED to converge to statistical equilibrium after {it}'
                f' iterations (took {duration:.2f} s).'
            )
            print(line)
        if return_final_convergence:
            finalConvergence = [J_update, pops_update]
            if prd and dRhoUpdate is not None:
                finalConvergence.append(dRhoUpdate)
            return it, finalConvergence
        else:
            return it
