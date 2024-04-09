# GridCal
# Copyright (C) 2015 - 2024 Santiago Peñate Vera
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
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import GridCalEngine.api as gce

from GridCalEngine.basic_structures import Vec, CscMat, CxVec, IntVec
import GridCalEngine.Simulations.PowerFlow.NumericalMethods.common_functions as cf
from GridCalEngine.Simulations.PowerFlow.NumericalMethods.ac_jacobian import AC_jacobian
from GridCalEngine.Utils.NumericalMethods.common import ConvexFunctionResult, ConvexMethodResult
from GridCalEngine.Utils.NumericalMethods.newton_raphson import newton_raphson
from GridCalEngine.Utils.NumericalMethods.powell import powell_dog_leg
from GridCalEngine.Utils.NumericalMethods.levenberg_marquadt import levenberg_marquardt
from GridCalEngine.Utils.NumericalMethods.autodiff import calc_autodiff_jacobian
from GridCalEngine.enumerations import SolverType
from GridCalEngine.Topology.admittance_matrices import compute_passive_admittances
from GridCalEngine.Utils.Sparse.csc import diags


def var2x(Va: Vec, Vm: Vec, tau: Vec, m: Vec) -> Vec:
    """
    Compose the unknowns vector
    :param Va: Array of voltage angles for the PV and PQ nodes
    :param Vm: Array of voltage modules for the PQ nodes
    :param tau: Array of branch tap angles for the controlled branches
    :param m: Array of branch tap modules for the controlled branches
    :return: [Va | Vm | tau | m]
    """
    return np.r_[Va, Vm, tau, m]


def compute_g(V: CxVec,
              Ybus: CscMat,
              Yf: CscMat,
              Yt: CscMat,
              S0: CxVec,
              I0: CxVec,
              Y0: CxVec,
              Vm: Vec,
              m: Vec,
              tau: Vec,
              Cf,
              Ct,
              F: IntVec,
              T: IntVec,
              pq: IntVec,
              noslack: IntVec,
              pvr: IntVec,
              k_tau: IntVec,
              k_m: IntVec):
    """
    Compose the power flow function
    :param V:
    :param Ybus:
    :param S0:
    :param I0:
    :param Y0:
    :param Vm:
    :param m:
    :param tau:
    :param Cf:
    :param Ct:
    :param F: node_from indices from every branch
    :param T: node_to indices from every branch
    :param pq:
    :param noslack:
    :return:
    """

    # Formulation 1
    Sbus = S0 + np.conj(I0 + Y0 * Vm) * Vm

    Scalc = V * np.conj(Ybus @ V)

    Vf = V[F]
    Vt = V[T]
    If = Yf * V
    It = Yt * V
    Sf_calc = Vf * np.conj(If)
    St_calc = Vt * np.conj(It)

    Vf_branch = Vm[F]
    Vt_branch = Vm[T]
    If_branch = Yf * Vm
    It_branch = Yt * Vm
    Sf_branch = Vf_branch * np.conj(If_branch)
    St_branch = Vt_branch * np.conj(It_branch)

    dS = Sbus - Scalc
    dSf_branch = Sf_branch - Sf_calc
    dSt_branch = St_branch - St_calc

    g = np.r_[dS[noslack].real, dS[np.r_[pq, pvr]].imag, dSf_branch[k_m].imag, dSt_branch[k_m].imag, dSf_branch[
        k_tau].real]  # TODO: falta crear el índice i_m_vr

    return g


def x2var(x: Vec, n_noslack: int, n_pqpvr: int, n_k_tau: int, n_k_m: int) -> Tuple[Vec, Vec, Vec, Vec, Vec]:
    """
    get the physical variables from the unknowns vector
    :param x: vector of unknowns
    :param n_noslack: number of non slack nodes
    :param n_pqpvr: number of pq and pvr nodes
    :param n_k_tau: number of branches with controlled tap angle
    :param n_k_m: number of branches with controlled tap module
    :return: Va, Vm, tau, m
    """
    a = 0
    b = n_noslack

    Va = x[a:b]
    a += b
    b += n_pqpvr

    Vm = x[a:b]
    a += b
    b += n_k_tau

    tau = x[a:b]
    a += b
    b += n_k_m

    m = x[a:b]

    return Va, Vm, tau, m


def compute_gx(V: CxVec, Ybus: CscMat, pvpq: IntVec, pq: IntVec) -> CscMat:
    """
    Compute the Jacobian matrix of the power flow function
    :param V:
    :param Ybus:
    :param pvpq:
    :param pq:
    :return:
    """
    return AC_jacobian(Ybus, V, pvpq, pq)


def compute_gx_autodiff(x: Vec,
                        # these are the args:
                        Va0: Vec,
                        Vm0: Vec,
                        Ybus: CscMat,
                        Yf: CscMat,
                        Yt: CscMat,
                        S0: CxVec,
                        I0: CxVec,
                        Y0: CxVec,
                        m: Vec,
                        tau: Vec,
                        Cf,
                        Ct,
                        F: IntVec,
                        T: IntVec,
                        pq: IntVec,
                        pvpq: IntVec,
                        pvr: IntVec,
                        pqpvr: IntVec,
                        k_tau: IntVec,
                        k_m: IntVec) -> ConvexFunctionResult:
    """

    :param x: vector of unknowns (handled by the solver)
    :param Va0:
    :param Vm0:
    :param Ybus:
    :param S0:
    :param I0:
    :param Y0:
    :param m:
    :param tau:
    :param Cf:
    :param Ct:
    :param F:
    :param T:
    :param pq:
    :param pvpq:
    :return:
    """
    npvpq = len(pvpq)
    n_pqpvr = len(pqpvr)
    n_k_tau = len(k_tau)
    n_k_m = len(k_m)
    Va = Va0.copy()
    Vm = Vm0.copy()
    Va[pvpq], Vm[pqpvr], tau[k_tau], m[k_m] = x2var(x=x, n_noslack=npvpq, n_pqpvr=n_pqpvr, n_k_tau=n_k_tau, n_k_m=n_k_m)
    V = Vm * np.exp(1j * Va)

    g = compute_g(V=V, Ybus=Ybus, S0=S0, I0=I0, Y0=Y0, Vm=Vm, m=m, tau=tau, Cf=Cf, Ct=Ct, F=F, T=T, pq=pq, noslack=pvpq,
                  Yf=Yf, Yt=Yt, pvr=pvr, k_tau=k_tau, k_m=k_m)

    return g


# TODO: hemos cambiado el nombre de la función porque la estaba uniendo a otras pf_function de otros power_flow_research
def pf_function2(x: Vec,
                 compute_jac: bool,
                 # these are the args:
                 Va0: Vec,
                 Vm0: Vec,
                 Ybus: CscMat,
                 Yf: CscMat,
                 Yt: CscMat,
                 S0: CxVec,
                 I0: CxVec,
                 Y0: CxVec,
                 m0: Vec,
                 tau0: Vec,
                 Cf: CscMat,
                 Ct: CscMat,
                 F: IntVec,
                 T: IntVec,
                 pq: IntVec,
                 noslack: IntVec,
                 pvr: IntVec,
                 pqpvr: IntVec,
                 k_tau: IntVec,
                 k_m: IntVec) -> ConvexFunctionResult:
    """

    :param x: vector of unknowns (handled by the solver)
    :param compute_jac: compute the jacobian? (handled by the solver)
    :param Va0:
    :param Vm0:
    :param Ybus:
    :param Yf:
    :param Yt
    :param S0:
    :param I0:
    :param Y0:
    :param m:
    :param tau:
    :param Cf:
    :param Ct:
    :param F:
    :param T:
    :param pq:
    :param noslack:
    :return:
    """
    nnoslack = len(noslack)
    npqpvr = len(pqpvr)
    n_k_tau = len(k_tau)
    n_k_m = len(k_m)
    Va = Va0.copy()
    Vm = Vm0.copy()
    tau = tau0.copy()
    m = m0.copy()
    Va[noslack], Vm[pqpvr], tau[k_tau], m[k_m] = x2var(x=x, n_noslack=nnoslack, n_pqpvr=npqpvr, n_k_tau=n_k_tau,
                                                       n_k_m=n_k_m)

    V = Vm * np.exp(1j * Va)

    g = compute_g(V=V, Ybus=Ybus, S0=S0, I0=I0, Y0=Y0, Vm=Vm, m=m, tau=tau, Cf=Cf, Ct=Ct, F=F, T=T, pq=pq,
                  noslack=noslack, pvr=pvr, Yf=Yf, Yt=Yt, k_tau=k_tau, k_m=k_m)

    if compute_jac:
        # Gx = compute_gx(V=V, Ybus=Ybus, pvpq=pvpq, pq=pq)

        Gx = calc_autodiff_jacobian(func=compute_gx_autodiff,
                                    x=x,
                                    arg=(Va0, Vm0, Ybus, Yf, Yt, S0, I0, Y0, m, tau, Cf, Ct, F, T, pq, noslack, pqpvr,
                                         pvr, k_tau, k_m))

    else:
        Gx = None

    return ConvexFunctionResult(f=g, J=Gx)


def run_pf(grid: gce.MultiCircuit, pf_options: gce.PowerFlowOptions):
    """

    :param grid:
    :param pf_options:
    :return:
    """
    nc = gce.compile_numerical_circuit_at(grid, t_idx=None)

    adm = compute_passive_admittances(R=nc.branch_data.R,
                                      X=nc.branch_data.X,
                                      G=nc.branch_data.G,
                                      B=nc.branch_data.B,
                                      vtap_f=nc.branch_data.virtual_tap_f,
                                      vtap_t=nc.branch_data.virtual_tap_t,
                                      Cf=nc.branch_data.C_branch_bus_f.tocsc(),
                                      Ct=nc.branch_data.C_branch_bus_t.tocsc(),
                                      Yshunt_bus=nc.Yshunt_from_devices,
                                      conn=nc.branch_data.conn,
                                      seq=1,
                                      add_windings_phase=False)
    Ybus = adm.Ybus
    pq = nc.pq
    pvpq = np.r_[nc.pv, nc.pq]
    k_tau = nc.k_tau
    n_k_tau = len(k_tau)
    k_m = nc.k_m
    n_k_m = len(k_m)
    pqpvr = np.r_[nc.pq, nc.pvr]
    pvr = nc.pvr
    npqpvr = len(pqpvr)
    npvpq = len(pvpq)
    S0 = nc.Sbus
    I0 = nc.Ibus
    Y0 = nc.YLoadBus
    m = nc.branch_data.tap_module
    tau = nc.branch_data.tap_angle
    Yf = nc.Yf
    Yt = nc.Yt
    Cf = nc.branch_data.C_branch_bus_f
    Ct = nc.branch_data.C_branch_bus_t
    F = nc.F
    T = nc.T
    Vm0 = np.abs(nc.Vbus)
    Va0 = np.angle(nc.Vbus)
    x0 = var2x(Va=Va0[pvpq], Vm=Vm0[pq], tau=tau[k_tau], m=m[k_m])

    logger = gce.Logger()

    if pf_options.solver_type == SolverType.NR:
        ret: ConvexMethodResult = newton_raphson(func=pf_function2,
                                                 func_args=(Va0, Vm0, Ybus, Yf, Yt, S0, Y0, I0, m, tau, Cf, Ct, F, T,
                                                            pq, pvpq, pvr, pqpvr, k_tau, k_m),
                                                 x0=x0,
                                                 tol=pf_options.tolerance,
                                                 max_iter=pf_options.max_iter,
                                                 trust=pf_options.trust_radius,
                                                 verbose=pf_options.verbose,
                                                 logger=logger)

    elif pf_options.solver_type == SolverType.PowellDogLeg:
        ret: ConvexMethodResult = powell_dog_leg(func=pf_function,
                                                 func_args=(Va0, Vm0, Ybus, S0, Y0, I0, m, tau, Cf, Ct, F, T, pq, pvpq),
                                                 x0=x0,
                                                 tol=pf_options.tolerance,
                                                 max_iter=pf_options.max_iter,
                                                 trust_region_radius=pf_options.trust_radius,
                                                 verbose=pf_options.verbose,
                                                 logger=logger)

    elif pf_options.solver_type == SolverType.LM:
        ret: ConvexMethodResult = levenberg_marquardt(func=pf_function,
                                                      func_args=(
                                                          Va0, Vm0, Ybus, S0, Y0, I0, m, tau, Cf, Ct, F, T, pq, pvpq),
                                                      x0=x0,
                                                      tol=pf_options.tolerance,
                                                      max_iter=pf_options.max_iter,
                                                      verbose=pf_options.verbose,
                                                      logger=logger)

    else:
        raise Exception(f"Solver not implemented {pf_options.solver_type.value}")

    Va = Va0.copy()
    Vm = Vm0.copy()
    Va[pvpq], Vm[pqpvr], tau[k_tau], m[k_m] = x2var(x=ret.x, n_noslack=npvpq, n_pqpvr=npqpvr, n_k_tau=n_k_tau, n_k_m=n_k_m)

    df = pd.DataFrame(data={"Vm": Vm, "Va": Va})
    print(df)

    print("Info:")
    ret.print_info()

    print("Logger:")
    logger.print()
    ret.plot_error()

    plt.show()


if __name__ == '__main__':
    import os

    # grid_ = linn5bus_example()

    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', '2869 Pegase.gridcal')
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', '1951 Bus RTE.xlsx')
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "GB Network.gridcal")
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "Iwamoto's 11 Bus.xlsx")
    fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "case14.m")
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "Illinois 200 Bus.gridcal")
    grid_ = gce.open_file(fname)

    pf_options_ = gce.PowerFlowOptions(solver_type=gce.SolverType.NR,
                                       max_iter=50,
                                       trust_radius=5.0,
                                       tolerance=1e-6,
                                       verbose=0)
    run_pf(grid=grid_, pf_options=pf_options_)
