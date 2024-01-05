# GridCal
# Copyright (C) 2015 - 2023 Santiago Peñate Vera
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# import collections
#
# collections.Callable = collections.abc.Callable

from typing import Callable, Tuple
import numpy as np
from scipy.sparse import csc_matrix as csc
from scipy import sparse
import timeit
from GridCalEngine.Utils.MIPS.ipm_test import NLP_test
from GridCalEngine.basic_structures import Vec, Mat
from GridCalEngine.Utils.Sparse.csc import pack_3_by_4, diags


def step_calculation(V: Vec, dV: Vec, NI: int):
    """
    This function calculates for each Lambda multiplier or its associated Slack variable
    the maximum allowed step in order to not violate the KKT condition Lambda > 0 and S > 0
    :param V: Array of multipliers or slack variables
    :param dV: Variation calculated in the Newton step
    :param NI: Number of inequalities.
    :return:
    """
    alpha = 1.0

    for i in range(NI):
        if dV[i] < 0:
            alpha = min(alpha, -V[i] / dV[i])

    alpha = min(0.99995 * alpha, 1.0)

    return alpha


def solver(x0: Vec,
           n_x: int,
           n_eq: int,
           n_ineq: int,
           func: Callable[[csc, csc, csc, csc, csc, Vec, Vec, Vec, csc, csc, csc, csc, float], Tuple[float, Vec, Vec, Vec, csc, csc, csc, csc, csc]],
           arg=(),
           gamma0=100,
           max_iter=100,
           verbose: int = 0):
    """
    Solve a non-linear problem of the form:

        min: f(x)
        s.t.
            G(x)  = 0
            H(x) <= 0
            xmin <= x <= xmax

    The problem is specified by a function `f_eval`
    This function is called with (x, lambda, pi) and
    returns (f, G, H, fx, Gx, Hx, fxx, Gxx, Hxx)

    where:
        x: array of variables
        lambda: Lagrange Multiplier associated with the inequality constraints
        pi: Lagrange Multiplier associated with the equality constraints
        f: objective function value (float)
        G: Array of equality mismatches (vec)
        H: Array of inequality mismatches (vec)
        fx: jacobian of f(x) (vec)
        Gx: Jacobian of G(x) (CSC mat)
        Hx: Jacobian of H(x) (CSC mat)
        fxx: Hessian of f(x) (CSC mat)
        Gxx: Hessian of G(x) (CSC mat)
        Hxx: Hessian of H(x) (CSC mat)

    :param x0: Initial solution
    :param n_x: Number of variables (size of x)
    :param n_eq: Number of equality constraints (rows of H)
    :param n_ineq: Number of inequality constraints (rows of G)
    :param func: A function pointer called with (x, lambda, pi, *args) that returns (f, G, H, fx, Gx, Hx, fxx, Gxx, Hxx)
    :param arg: Tuple of arguments to call func: func(x, lambda, pi, *arg)
    :param gamma0:
    :param max_iter:
    :param verbose:
    :return:
    """
    START = timeit.default_timer()

    # Init iteration values
    error = 1e20
    iter_counter = 0
    f = 0.0  # objective function
    x = x0.copy()
    gamma = gamma0

    # Init multiplier values. Defaulted at 1.
    pi_vec = np.ones(n_eq)
    lmbda_vec = np.ones(n_ineq)
    T = np.ones(n_ineq)
    E = np.ones(n_ineq)
    inv_T = diags(1.0 / T)
    lmbda_mat = diags(lmbda_vec)

    while error > gamma and iter_counter < max_iter:

        # Evaluate the functions, gradients and hessians at the current iteration.
        f, G, H, fx, Gx, Hx, fxx, Gxx, Hxx = func(x, lmbda_vec, pi_vec, *arg)

        # compose the Jacobian
        M = fxx + Gxx + Hxx + Hx @ inv_T @ lmbda_mat @ Hx.T
        J = pack_3_by_4(M, Gx.tocsc(), Gx.T)

        # compose the residual
        N = fx + Hx @ lmbda_vec + Hx @ inv_T @ (gamma * E + lmbda_mat @ H) + Gx @ pi_vec
        r = - np.r_[N, G]

        # Find the reduced problem residuals and split them
        dXP = sparse.linalg.spsolve(J, r)
        dX = dXP[:n_x]
        dP = dXP[n_x:]

        # Calculate the inequalities residuals using the reduced problem residuals
        dT = -H - T - Hx.T @ dX
        dL = -lmbda_vec + inv_T @ (gamma * E - lmbda_mat @ dT)

        # Compute the maximum step allowed
        alphap = step_calculation(T, dT, n_ineq)
        alphad = step_calculation(lmbda_vec, dL, n_ineq)

        # Update the values of the variables and multipliers
        x += dX * alphap
        T += dT * alphap
        lmbda_vec += dL * alphad
        pi_vec += dP * alphad

        inv_T = diags(1.0 / T)
        lmbda_mat = diags(lmbda_vec)

        # Compute the maximum error and the new gamma value
        error = np.max([np.max(abs(dXP)), np.max(abs(dL)), np.max(abs(dT))])

        # newgamma = 0.5 * gamma
        newgamma = 0.1 * (T @ lmbda_vec) / n_ineq
        gamma = max(newgamma, 1e-5)  # Maximum tolerance requested.

        # Add an iteration step
        iter_counter += 1

        if verbose > 1:
            print(f'Iteration: {iter_counter}', "-" * 80)
            print("\tx:", x)
            print("\tGamma:", gamma)
            print("\tErr:", error)

    END = timeit.default_timer()

    if verbose > 0:
        print(f'SOLUTION', "-" * 80)
        print("\tx:", x)
        print("\tF.obj:", f)
        print("\tErr:", error)
        print(f'Iterations: {iter_counter}')
        print('\tTime elapsed (s): ', END - START)

    return x, error, gamma


def test_solver():
    X = np.array([2., 1.1, 0.])
    solver(x0=X, n_x=3, n_eq=1, n_ineq=2, func=NLP_test, arg=(), verbose=1)

    return


if __name__ == '__main__':
    test_solver()
