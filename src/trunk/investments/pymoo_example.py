from GridCal.ThirdParty.pymoo.algorithms.moo.nsga2 import NSGA2
from GridCal.ThirdParty.pymoo.algorithms.moo.nsga3 import NSGA3
from GridCal.ThirdParty.pymoo.problems import get_problem
from GridCal.ThirdParty.pymoo.optimize import minimize
from GridCal.ThirdParty.pymoo.visualization.scatter import Scatter

problem = get_problem("zdt1")

algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 80),
               seed=1,
               verbose=True)

plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()