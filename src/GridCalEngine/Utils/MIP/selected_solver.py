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

"""
Uncomment the inteface to use: Pulp or OrTools
"""

from GridCalEngine.Utils.MIP.ortools import (LpExp, LpVar, LpModel, get_lp_var_value, lpDot,
                                             get_available_mip_solvers, set_var_bounds)

# from GridCalEngine.Utils.MIP.pulp import (LpExp, LpVar, LpModel, get_lp_var_value, lpDot,
#                                           get_available_mip_solvers, set_var_bounds)
