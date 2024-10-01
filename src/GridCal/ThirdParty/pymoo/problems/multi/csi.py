import GridCal.ThirdParty.pymoo.gradient.toolbox as anp
import numpy as np

from GridCal.ThirdParty.pymoo.core.problem import Problem


class CSI(Problem):
    def __init__(self):
        xl = np.array([0.500, 0.450, 0.500, 0.500, 0.875, 0.400, 0.400])
        xu = np.array([1.500, 1.350, 1.500, 1.500, 2.625, 1.200, 1.200])
        super().__init__(n_var=7, n_obj=3, n_ieq_constr=10, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):

        # the definition is index 1 based -> simply add a dummy var in the beginning
        x = anp.column_stack([anp.zeros((len(x), 1)), x])

        F = 4.72 - 0.5 * x[:, 4] - 0.19 * x[:, 2] * x[:, 3]

        V_mbp = 10.58 - 0.674 * x[:, 1] * x[:, 2] - 0.67275 * x[:, 2]

        V_fd = 16.45 - 0.489 * x[:, 3] * x[:, 7] - 0.843 * x[:, 5] * x[:, 6]

        f1 = 1.98 + 4.9 * x[:, 1] + 6.67 * x[:, 2] + 6.98 * x[:, 3] + 4.01 * x[:, 4] + 1.78 * x[:, 5] + \
             0.00001 * x[:, 6] + 2.73 * x[:, 7]

        f2 = F

        f3 = 0.5 * (V_mbp + V_fd)

        g1 = 1.16 - 0.3717 * x[:, 2] * x[:, 4] - 0.0092928 * x[:, 3] - 1

        g2 = 0.261 - 0.0159 * x[:, 1] * x[:, 2] - 0.06486 * x[:, 1] - 0.019 * x[:, 2] * x[:, 7] + \
             0.0144 * x[:, 3] * x[:, 5] + 0.0154464 * x[:, 6] - 0.32

        g3 = 0.214 + 0.00817 * x[:, 5] - 0.045195 * x[:, 1] - 0.0135168 * x[:, 1] + 0.03099 * x[:, 2] * x[:, 6] - \
             0.018 * x[:, 2] * x[:, 7] + 0.007176 * x[:, 3] + 0.023232 * x[:, 3] - 0.00364 * x[:, 5] * x[:, 6] - \
             0.018 * x[:, 2] ** 2 - 0.32

        g4 = 0.74 - 0.61 * x[:, 2] - 0.031296 * x[:, 3] - 0.031872 * x[:, 7] + 0.227 * x[:, 2] ** 2 - 0.32

        g5 = 28.98 + 3.818 * x[:, 3] - 4.2 * x[:, 1] * x[:, 2] + 1.27296 * x[:, 6] - 2.68065 * x[:, 7] - 32

        g6 = 33.86 + 2.95 * x[:, 3] - 5.057 * x[:, 1] * x[:, 2] - 3.795 * x[:, 2] - 3.4431 * x[:, 7] + 1.45728 - 32

        g7 = 46.36 - 9.9 * x[:, 2] - 4.4505 * x[:, 1] - 32

        g8 = F - 4

        g9 = V_mbp - 9.9

        g10 = V_fd - 15.7

        out["F"] = anp.column_stack([f1, f2, f3])
        out["G"] = anp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10])
