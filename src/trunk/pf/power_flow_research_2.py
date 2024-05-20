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
from GridCalEngine.Topology.simulation_indices import SimulationIndicesV2
from GridCalEngine.Utils.Sparse.csc import diags


def linn5bus_multislack():
    """
    Grid from Lynn Powel's book
    """
    # declare a circuit object
    grid = gce.MultiCircuit()

    # Add the buses and the generators and loads attached
    bus1 = gce.Bus('Bus 1', vnom=20)
    bus1.is_slack = True  # we may mark the bus a slack
    grid.add_bus(bus1)

    # add a generator to the bus 1
    gen1 = gce.Generator('Slack Generator', vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=1000)

    grid.add_generator(bus1, gen1)

    # add bus 2 with a load attached
    bus2 = gce.Bus('Bus 2', vnom=20)
    grid.add_bus(bus2)
    grid.add_load(bus2, gce.Load('load 2', P=40, Q=20))

    # add bus 3 with a load attached
    bus3 = gce.Bus('Bus 3', vnom=20)
    grid.add_bus(bus3)
    grid.add_load(bus3, gce.Load('load 3', P=25, Q=15))

    # add bus 4 with a load attached
    bus4 = gce.Bus('Bus 4', vnom=20)
    grid.add_bus(bus4)
    grid.add_load(bus4, gce.Load('load 4', P=40, Q=20))

    # bus5
    bus5 = gce.Bus('Bus 5', vnom=20)
    grid.add_bus(bus5)

    # add bus 6 with a generator
    bus6 = gce.Bus('Bus 6', vnom=20)
    bus6.is_slack = True
    grid.add_bus(bus6)

    # add bus5 with a generator
    gen3 = gce.Generator('Generator 3', control_bus=bus5, vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=400)
    grid.add_generator(bus5, gen3)

    gen2 = gce.Generator('Generator 2', control_bus=bus5, vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=4000)

    grid.add_generator(bus6, gen2)

    # TODO: ¿Medidas para saber si es from o to?
    # add Lines connecting the buses
    line_1_2 = gce.Line(bus1, bus2, name='line 1-2', r=0.05, x=0.11, b=0.02, rate=1000)
    grid.add_line(line_1_2)
    line_1_3 = gce.Line(bus1, bus3, name='line 1-3', r=0.05, x=0.11, b=0.02, rate=1000)
    grid.add_line(line_1_3)
    line_1_5 = gce.Line(bus1, bus5, name='line 1-5', r=0.03, x=0.08, b=0.02, rate=1000)
    grid.add_line(line_1_5)
    line_2_3 = gce.Line(bus2, bus3, name='line 2-3', r=0.04, x=0.09, b=0.02, rate=1000)
    grid.add_line(line_2_3)
    line_2_5 = gce.Line(bus2, bus5, name='line 2-5', r=0.04, x=0.09, b=0.02, rate=1000)
    grid.add_line(line_2_5)
    line_3_4 = gce.Line(bus3, bus4, name='line 3-4', r=0.06, x=0.13, b=0.03, rate=1000)
    grid.add_line(line_3_4)
    line_5_6 = gce.Line(bus5, bus6, name='line 5-6', r=0.06, x=0.13, b=0.03, rate=1000)
    grid.add_line(line_5_6)
    grid.add_transformer2w(gce.Transformer2W(bus4, bus5, name='transformer 4-5', r=0.04, x=0.09, b=0.02, rate=1000,
                                             tap_module_control_mode=gce.TapModuleControl.Vm,
                                             regulation_bus=bus3
                                             ))
    #grid.add_transformer2w(gce.Transformer2W(bus4, bus5, name='transformer 4-5', r=0.04, x=0.09, b=0.02, rate=1000,
    #                                         tap_module_control_mode=gce.TapModuleControl.Qt,
    #                                         regulation_branch=line_5_6
    #                                         ))
    #grid.add_transformer2w(gce.Transformer2W(bus4, bus5, name='transformer 4-5', r=0.04, x=0.09, b=0.02, rate=1000,
    #                                         tap_angle_control_mode=gce.TapAngleControl.Pt,
    #                                         regulation_branch=line_5_6
    #                                         ))

    return grid


def linn5bus_example():
    """
    Grid from Lynn Powel's book
    """
    # declare a circuit object
    grid = gce.MultiCircuit()

    # Add the buses and the generators and loads attached
    bus1 = gce.Bus('Bus 1', vnom=20)
    # bus1.is_slack = True  # we may mark the bus a slack
    grid.add_bus(bus1)

    # add a generator to the bus 1
    gen1 = gce.Generator('Slack Generator', vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=1000)

    grid.add_generator(bus1, gen1)

    # add bus 2 with a load attached
    bus2 = gce.Bus('Bus 2', vnom=20)
    grid.add_bus(bus2)
    grid.add_load(bus2, gce.Load('load 2', P=40, Q=20))

    # add bus 3 with a load attached
    bus3 = gce.Bus('Bus 3', vnom=20)
    grid.add_bus(bus3)
    grid.add_load(bus3, gce.Load('load 3', P=25, Q=15))

    # add bus 4 with a load attached
    bus4 = gce.Bus('Bus 4', vnom=20)
    grid.add_bus(bus4)
    grid.add_load(bus4, gce.Load('load 4', P=40, Q=20))

    # bus5
    bus5 = gce.Bus('Bus 5', vnom=20)
    grid.add_bus(bus5)

    # add bus 6 with a generator
    bus6 = gce.Bus('Bus 6', vnom=20)
    # bus6.is_slack = True
    grid.add_bus(bus6)

    # add bus 5 with a load attached
    # grid.add_load(bus5, gce.Load('load 5', P=50, Q=20))

    # add bus5 with a generator
    gen3 = gce.Generator('Generator 3', control_bus=bus5, vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=400)
    grid.add_generator(bus5, gen3)

    gen2 = gce.Generator('Generator 2', control_bus=bus5, vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=400)

    grid.add_generator(bus6, gen2)

    # TODO: ¿Medidas para saber si es from o to?
    # add Lines connecting the buses
    grid.add_line(gce.Line(bus1, bus2, name='line 1-2', r=0.05, x=0.11, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus1, bus3, name='line 1-3', r=0.05, x=0.11, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus1, bus5, name='line 1-5', r=0.03, x=0.08, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus2, bus3, name='line 2-3', r=0.04, x=0.09, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus2, bus5, name='line 2-5', r=0.04, x=0.09, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus3, bus4, name='line 3-4', r=0.06, x=0.13, b=0.03, rate=1000))
    grid.add_line(gce.Line(bus5, bus6, name='line 5-6', r=0.06, x=0.13, b=0.03, rate=1000))
    grid.add_transformer2w(gce.Transformer2W(bus4, bus5, name='transformer 4-5', r=0.04, x=0.09, b=0.02, rate=1000,
                                             tap_module_control_mode=gce.TapModuleControl.Vm,
                                             regulation_bus=bus2
                                             ))

    return grid


def linn5bus_us():
    """
    Grid from Lynn Powel's book
    """
    # declare a circuit object
    grid = gce.MultiCircuit()

    # Add the buses and the generators and loads attached
    bus1 = gce.Bus('Bus 1', vnom=138)
    # bus1.is_slack = True  # we may mark the bus a slack
    grid.add_bus(bus1)
    grid.add_load(bus1, gce.Load('load 1', P=1000, Q=250))

    # add bus 2 with a load attached
    bus2 = gce.Bus('Bus 2', vnom=138)
    grid.add_bus(bus2)
    grid.add_load(bus2, gce.Load('load 2', P=1000, Q=250))
    grid.add_shunt(bus2, gce.Shunt('BC 1', B=2.0))

    # add bus 3 with a load attached
    bus3 = gce.Bus('Bus 3', vnom=138)
    grid.add_bus(bus3)

    # add bus 4 with a load attached
    bus4 = gce.Bus('Bus 4', vnom=138)
    grid.add_bus(bus4)
    gen1 = gce.Generator('Generator 1', control_bus=bus4, vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=1000)
    grid.add_generator(bus4, gen1)

    # bus5
    bus5 = gce.Bus('Bus 5', vnom=20)
    grid.add_bus(bus5)
    bus5.is_slack = True
    gen2 = gce.Generator('Generator Slack', control_bus=bus5, vset=1.0, Pmin=0, Pmax=1000,
                         Qmin=-1000, Qmax=1000, Cost=15, Cost2=0.0, Snom=400)
    grid.add_generator(bus5, gen2)

    # tr1 = gce.Transformer2W(bus2, bus4, name='transformer 2-4', r=0.001, x=0.01, b=0.0, rate=1000,
    #                  tap_module_max=1.1, tap_module_min=0.9, tap_module=1.0,
    #                  tap_module_control_mode=gce.TapModuleControl.Vm,
    #                  regulation_bus=bus4)

    tr1 = gce.Transformer2W(bus_from=bus2, bus_to=bus4, name='transformer 2-4', r=0.001, x=0.01, b=0.0, rate=1000,
                            tap_phase_max=1.7, tap_phase_min=1.7, tap_module=1.0,
                            tap_angle_control_mode=gce.TapAngleControl.Pf, Pset=1200,
                            regulation_branch=None)

    tr1.tap_changer = gce.TapChanger(total_positions=21, neutral_position=11, dV=0.01)

    tr2 = gce.Transformer2W(bus_from=bus1, bus_to=bus3, name='transformer 1-3', r=0.001, x=0.015, b=0.0, rate=1000,
                            tap_phase_max=3.0, tap_phase_min=-3.0, tap_module=1.0,
                            tap_angle_control_mode=gce.TapAngleControl.Pf, Pset=1200,
                            regulation_branch=None)
    tr2.tap_changer = gce.TapChanger(total_positions=21, neutral_position=11, dV=0.01)

    # add Lines connecting the buses
    grid.add_line(gce.Line(bus_from=bus1, bus_to=bus2, name='line 1-2', r=0.01, x=0.01, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus_from=bus3, bus_to=bus4, name='line 3-4', r=0.005, x=0.02, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus_from=bus4, bus_to=bus5, name='line 4-5', r=0.005, x=0.02, b=0.02, rate=1000))
    grid.add_line(gce.Line(bus_from=bus5, bus_to=bus3, name='line 5-3', r=0.005, x=0.01, b=0.02, rate=1000))
    grid.add_transformer2w(tr1)
    grid.add_transformer2w(tr2)

    return grid


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
              Sf0: CxVec,
              St0: CxVec,
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

    dS = Sbus - Scalc
    dSf_branch = Sf0 - Sf_calc
    dSt_branch = St0 - St_calc

    g = np.r_[
        dS[noslack].real,  # dP
        dS[np.r_[pq, pvr]].imag,  # dQ
        dSf_branch[k_m].imag,  # dQf
        dSt_branch[k_m].imag,  # dQt
        dSf_branch[k_tau].real  # dPf
    ]  # TODO: falta crear el índice i_m_vr

    return g


def x2var(x: Vec, n_noslack: int, n_pqpvr: int, n_k_tau: int, n_k_m: int) -> Tuple[Vec, Vec, Vec, Vec]:
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
                        Sf0: CxVec,
                        St0: CxVec,
                        m: Vec,
                        tau: Vec,
                        Cf,
                        Ct,
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
    :param noslack:
    :return:
    """
    npvpq = len(noslack)
    n_pqpvr = len(pqpvr)
    n_k_tau = len(k_tau)
    n_k_m = len(k_m)
    Va = Va0.copy()
    Vm = Vm0.copy()
    Va[noslack], Vm[pqpvr], tau[k_tau], m[k_m] = x2var(x=x, n_noslack=npvpq, n_pqpvr=n_pqpvr, n_k_tau=n_k_tau,
                                                       n_k_m=n_k_m)
    V = Vm * np.exp(1j * Va)

    g = compute_g(V=V, Ybus=Ybus, S0=S0, I0=I0, Y0=Y0, Vm=Vm, m=m, tau=tau, Cf=Cf, Ct=Ct, F=F, T=T, pq=pq,
                  noslack=noslack,
                  Yf=Yf, Yt=Yt, pvr=pvr, k_tau=k_tau, k_m=k_m, Sf0=Sf0, St0=St0)

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
                 Sf0: CxVec,
                 St0: CxVec,
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
                  noslack=noslack, pvr=pvr, Yf=Yf, Yt=Yt, k_tau=k_tau, k_m=k_m, Sf0=Sf0, St0=St0)
    # TODO: Reminder: es posible que a lo largo de las iteraciones cambien los tipos de nudos y haya que actualizar la
    # prioridad de algún control
    if compute_jac:
        # Gx = compute_gx(V=V, Ybus=Ybus, pvpq=pvpq, pq=pq)

        Gx = calc_autodiff_jacobian(func=compute_gx_autodiff,
                                    x=x,
                                    arg=(Va0, Vm0, Ybus, Yf, Yt, S0, I0, Y0, Sf0, St0, m, tau, Cf, Ct, F, T,
                                         pq, noslack, pvr, pqpvr, k_tau, k_m))

    else:
        Gx = None

    return ConvexFunctionResult(f=g, J=Gx)


def indices_computation(grid: gce.MultiCircuit):
    """
    :param grid
    :return:
    """

    nc = gce.compile_numerical_circuit_at(grid, t_idx=None)

    indices = SimulationIndicesV2(
        bus_types=nc.bus_types,
        Pbus=nc.Sbus.real,
        branch_control_bus=nc.branch_data.ctrl_bus,
        branch_control_branch=nc.branch_data.ctrl_branch,
        branch_control_mode_m=nc.branch_data.ctrl_mode_m,
        branch_control_mode_tau=nc.branch_data.ctrl_mode_tau,
        generator_control_bus=nc.generator_data.ctrl_bus,
        generator_iscontrolled=nc.generator_data.controllable,
        generator_buses=nc.generator_data.genbus,
        F=nc.F,
        T=nc.T,
        dc=nc.dc_indices
    )
    ref, pq, pv, pvr, no_slack, pqv, k_m_vr, k_m_qf, k_m_qt, k_tau_pf, k_tau_pt = indices.compute_indices(
                                                    Pbus=nc.Sbus.real,
                                                    types=nc.bus_types,
                                                    generator_control_bus=nc.generator_data.ctrl_bus,
                                                    generator_buses=nc.generator_data.genbus,
                                                    branch_control_bus=nc.branch_data.ctrl_bus,
                                                    branch_control_branch=nc.branch_data.ctrl_branch,
                                                    Snomgen=nc.generator_data.snom,
                                                    branch_control_mode_m=nc.branch_data.ctrl_mode_m,
                                                    branch_control_mode_tau=nc.branch_data.ctrl_mode_tau)

    return ref, pq, pv, pvr, no_slack, pqv, k_m_vr, k_m_qf, k_m_qt, k_tau_pf, k_tau_pt


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

    indices = SimulationIndicesV2(
        bus_types=nc.bus_types,
        Pbus=nc.Sbus.real,
        branch_control_bus=nc.branch_data.ctrl_bus,
        branch_control_branch=nc.branch_data.ctrl_branch,
        branch_control_mode_m=nc.branch_data.ctrl_mode_m,
        branch_control_mode_tau=nc.branch_data.ctrl_mode_tau,
        generator_control_bus=nc.generator_data.ctrl_bus,
        generator_iscontrolled=nc.generator_data.controllable,
        generator_buses=nc.generator_data.genbus,
        F=nc.F,
        T=nc.T,
        dc=nc.dc_indices
    )
    ref, pq, pv, pvr, no_slack, pqv, k_m_vr = indices.compute_indices(Pbus=nc.Sbus.real,
                                                    types=nc.bus_types,
                                                    generator_control_bus=nc.generator_data.ctrl_bus,
                                                    generator_buses=nc.generator_data.genbus)

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
    Sf0 = nc.branch_data.Pfset + 1j * nc.branch_data.Qfset
    St0 = nc.branch_data.Ptset + 1j * nc.branch_data.Qtset
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
                                                 func_args=(
                                                 Va0, Vm0, Ybus, Yf, Yt, S0, Y0, I0, Sf0, St0, m, tau, Cf, Ct, F, T,
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
    Va[pvpq], Vm[pqpvr], tau[k_tau], m[k_m] = x2var(x=ret.x, n_noslack=npvpq, n_pqpvr=npqpvr, n_k_tau=n_k_tau,
                                                    n_k_m=n_k_m)

    df = pd.DataFrame(data={"Vm": Vm, "Va": Va})
    print(df)

    print("Info:")
    ret.print_info()

    print("Logger:")
    logger.print()
    ret.plot_error()

    plt.show()


def test_multiple_slack() -> None:
    gridtest_ = linn5bus_multislack()

    ref, pq, pv, pvr, no_slack, pqv, k_m_vr, k_m_qf, k_m_qt, k_tau_pf, k_tau_pt = indices_computation(grid=gridtest_)

    # check that it exists only one slack node
    assert (len(ref) == 1)

    # check that slack node is the one with higher nominal power
    candidate = gridtest_.generators[0]
    for g in gridtest_.generators:
        if g.Snom > candidate.Snom:
            candidate = g
    # TODO: hay que cambiar esto en otro punto diferente al del compute_indices
    # assert (candidate.bus == gridtest_.buses[ref[0]])



if __name__ == '__main__':
    import os

    # grid_ = linn5bus_example()

    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', '2869 Pegase.gridcal')
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', '1951 Bus RTE.xlsx')
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "GB Network.gridcal")
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "Iwamoto's 11 Bus.xlsx")
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', '5n2g.RAW')
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "case14.m")
    # fname = os.path.join('..', '..', '..', 'Grids_and_profiles', 'grids', "Illinois 200 Bus.gridcal")
    # grid_ = gce.open_file(fname)
    grid_ = linn5bus_example()

    pf_options_ = gce.PowerFlowOptions(solver_type=gce.SolverType.NR,
                                       max_iter=50,
                                       trust_radius=5.0,
                                       tolerance=1e-6,
                                       verbose=0)
    run_pf(grid=grid_, pf_options=pf_options_)
