import time
from typing import TYPE_CHECKING, Optional, Type

from .iteration_update import IterationUpdate

if TYPE_CHECKING:
    from . import Context

class ConvergenceCriteria:
    '''
    Abstract base class for determining convergence inside `iterate_ctx_se`. A
    derived variant of this class will be instantiated by `iterate_ctx_se`
    dependent upon its arguments. The default implementation is
    `DefaultConvergenceCriteria`.

    Parameters
    ----------
    ctx : Context
        The context being iterated.
    JTol : float
        The value of JTol passed to `iterate_ctx_se`.
    popsTol : float
        The value of popsTol passed to `iterate_ctx_se`.
    rhoTol : float or None
        The value of rhoTol passed to `iterate_ctx_se`.
    '''
    def __init__(self, ctx: 'Context', JTol: float, popsTol: float, rhoTol: Optional[float]):
        raise NotImplementedError

    def is_converged(self, JUpdate: IterationUpdate, popsUpdate: IterationUpdate,
                     prdUpdate: Optional[IterationUpdate]) -> bool:
        '''
        This function takes the IterationUpdate objects from
        `ctx.formal_sol_gamma_matrices` and `ctx.stat_equil` and optionally from
        `ctx.prd_redistribute` (or None).  Should return a bool indicated
        whether the Context is sufficiently converged.
        '''
        raise NotImplementedError


class DefaultConvergenceCriteria(ConvergenceCriteria):
    '''
    Default ConvergenceCriteria implementation. Usually sufficient for
    statistical equilibrium problems, but you may occasionally need to override
    this.

    Parameters
    ----------
    ctx : Context
        The context being iterated.
    JTol : float
        The value of JTol passed to `iterate_ctx_se`.
    popsTol : float
        The value of popsTol passed to `iterate_ctx_se`.
    rhoTol : float or None
        The value of rhoTol passed to `iterate_ctx_se`.
    '''

    def __init__(self, ctx: 'Context', JTol: float, popsTol: float, rhoTol: Optional[float]):
        self.ctx = ctx
        self.JTol = JTol
        self.popsTol = popsTol
        self.rhoTol = rhoTol

    def is_converged(self, JUpdate: IterationUpdate, popsUpdate: IterationUpdate,
                     prdUpdate: Optional[IterationUpdate]) -> bool:
        '''
        Returns whether the context is converged.
        '''
        updates = [JUpdate, popsUpdate]
        if prdUpdate is not None:
            updates.append(prdUpdate)

        terminate = True
        for update in updates:
            terminate = terminate and (update.dJMax < self.JTol)
            terminate = terminate and (update.dPopsMax < self.popsTol)
            if prdUpdate and self.rhoTol is not None:
                terminate = terminate and (update.dRhoMax < self.rhoTol)
        terminate = terminate and self.ctx.crswDone

        return terminate


def iterate_ctx_se(ctx: 'Context', Nscatter: int=3, NmaxIter: int=2000,
                   prd: bool=False, JTol: float=5e-3, popsTol: float=1e-3,
                   rhoTol: Optional[float]=None, prdIterTol: float=1e-2,
                   maxPrdSubIter: int=3, printInterval: float=0.2,
                   quiet: bool=False,
                   convergence: Optional[Type[ConvergenceCriteria]]=None,
                   returnFinalConvergence: bool=False):
    '''
    Iterate a configured Context towards statistical equilibrium solution.

    Parameters
    ----------
    ctx : Context
        The context to iterate.
    Nscatter : int, optional
        The number of lambda iterations to perform for an initial estimate of J
        (default: 3).
    NmaxIter : int, optional
        The maximum number of iterations (including Nscatter) to take (default:
        2000).
    prd: bool, optional
        Whether to perform PRD subiterations to estimate rho for PRD lines
        (default: False).
    JTol: float, optional
        The maximum relative change in J from one iteration to the next
        (default: 5e-3).
    popsTol: float, optional
        The maximum relative change in an atomic population from one iteration
        to the next (default: 1e-3).
    rhoTol: float, optional
        The maximum relative change in rho for a PRD line on the final
        subiteration from one iteration to the next. If None, the change in rho
        will not be considered in judging convergence (default: None).
    prdIterTol: float, optional
        The maximum relative change in rho for a PRD line below which PRD
        subiterations will cease for this iteration (default: 1e-2).
    maxPrdSubIter : int, optional
        The maximum number of PRD subiterations to make, whether or not rho has
        reached the tolerance of prdIterTol (which isn't necessary every
        iteration). (Default: 3)
    printInterval : float, optional
        The interval between printing the update size information in seconds. A
        value of 0.0 will print every iteration (default: 0.2).
    quiet : bool, optional
        Overrides any other print arguments and iterates silently if True.
        (Default: False).
    convergence : derived ConvergenceCriteria class, optional
        The ConvergenceCriteria version to be used in determining convergence.
        Will be instantiated by this function, and the `is_converged` method
        will then be used.  (Default: DefaultConvergenceCriteria).
    returnFinalConvergence : bool, optional
        Whether to return the IterationUpdate objects used in the final
        convergence decision, if True, these will be returned in a list as the
        second return value. (Default: False).

    Returns
    -------
    it : int
        The number of iterations taken.
    finalIterationUpdates : List[IterationUpdate], optional
        The final IterationUpdates computed, if requested by `returnFinalConvergence`.
    '''

    prevPrint = 0.0
    printNow = True
    alwaysPrint = (printInterval == 0.0)
    startTime = time.time()

    if convergence is None:
        convergence = DefaultConvergenceCriteria
    conv = convergence(ctx, JTol, popsTol, rhoTol)

    for it in range(NmaxIter):
        JUpdate : IterationUpdate = ctx.formal_sol_gamma_matrices()
        if (not quiet and
            (alwaysPrint or ((now := time.time()) >= prevPrint + printInterval))):
            printNow = True
            if not alwaysPrint:
                prevPrint = now

        if not quiet and printNow:
            print(f'-- Iteration {it}:')
            print(JUpdate.compact_representation())

        if it < Nscatter:
            if not quiet and printNow:
                print('    (Lambda iterating background)')
            # NOTE(cmo): reset print state
            printNow = False
            continue

        popsUpdate : IterationUpdate = ctx.stat_equil()
        if not quiet and printNow:
            print(popsUpdate.compact_representation())

        dRhoUpdate : Optional[IterationUpdate]
        if prd:
            dRhoUpdate = ctx.prd_redistribute(maxIter=maxPrdSubIter, tol=prdIterTol)
            if not quiet and printNow and dRhoUpdate is not None:
                print(dRhoUpdate.compact_representation())
        else:
            dRhoUpdate = None

        terminate = conv.is_converged(JUpdate, popsUpdate, dRhoUpdate)

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
                    print(JUpdate.compact_representation())
                    print(popsUpdate.compact_representation())
                    if prd and dRhoUpdate is not None:
                        print(dRhoUpdate.compact_representation())
                print(line)
                print(f'Context converged to statistical equilibrium in {it}'
                      f' iterations after {duration:.2f} s.')
                print(line)
            if returnFinalConvergence:
                finalConvergence = [JUpdate, popsUpdate]
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
            print(JUpdate.compact_representation())
            print(popsUpdate.compact_representation())
            if prd and dRhoUpdate is not None:
                print(dRhoUpdate.compact_representation())
            print(line)
            print(f'Context FAILED to converge to statistical equilibrium after {it}'
                  f' iterations (took {duration:.2f} s).')
            print(line)
        if returnFinalConvergence:
            finalConvergence = [JUpdate, popsUpdate]
            if prd and dRhoUpdate is not None:
                finalConvergence.append(dRhoUpdate)
            return it, finalConvergence
        else:
            return it
