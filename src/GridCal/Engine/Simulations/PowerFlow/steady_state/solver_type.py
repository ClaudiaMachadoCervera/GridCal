from enum import Enum


class SolverType(Enum):
    """
    Refer to the :ref:`Power Flow section<power_flow>` for details about the different
    algorithms supported by **GridCal**.
    """

    NR = 'Newton Raphson'
    NRFD_XB = 'Fast decoupled XB'
    NRFD_BX = 'Fast decoupled BX'
    GAUSS = 'Gauss-Seidel'
    DC = 'Linear DC'
    HELM = 'Holomorphic Embedding'
    HELM_Z_PV = 'HELM-Z-PV'
    HELM_Z_PQ = 'HELM-Z-PQ'
    HELM_CHENGXI_VANILLA = 'HELM-Chengxi-Vanilla'
    HELM_CHENGXI_2 = 'HELM-Chengxi-2'
    HELM_CHENGXI_CORRECTED = 'HELM-Chengxi-Corrected'
    HELM_PQ = 'HELM-PQ'
    HELM_VECT_ASU = 'HELM-Vect-ASU'
    HELM_WALLACE = 'HELM-Wallace'
    ZBUS = 'Z-Gauss-Seidel'
    IWAMOTO = 'Iwamoto-Newton-Raphson'
    CONTINUATION_NR = 'Continuation-Newton-Raphson'
    LM = 'Levenberg-Marquardt'
    FASTDECOUPLED = 'Fast decoupled'
    LACPF = 'Linear AC'
    DC_OPF = 'Linear DC OPF'
    AC_OPF = 'Linear AC OPF'
    NRI = 'Newton-Raphson in current'
    DYCORS_OPF = 'DYCORS OPF'
    GA_OPF = 'Genetic Algorithm OPF'
    NELDER_MEAD_OPF = 'Nelder Mead OPF'