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
from GridCalEngine.IO.base.units import UnitMultiplier, UnitSymbol
from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.operational_limit import OperationalLimit
from GridCalEngine.IO.cim.cgmes.cgmes_enums import cgmesProfile, SynchronousMachineKind, GeneratorControlSource, WindingConnection, ControlAreaTypeKind, SVCControlMode, BatteryStateKind, DCPolarityKind, FuelType, AsynchronousMachineKind, PhaseCode, Currency, HydroPlantStorageKind, HydroTurbineKind, CurveStyle, OperationalLimitDirectionKind, LimitKind, UnitSymbol, CsPpccControlKind, CsOperatingModeKind, HydroEnergyConversionKind, SynchronousMachineOperatingMode, DCConverterOperatingModeKind, UnitMultiplier, RegulatingControlModeKind


class VoltageLimit(OperationalLimit):
	def __init__(self, rdfid='', tpe='VoltageLimit'):
		OperationalLimit.__init__(self, rdfid, tpe)

		self.normalValue: float = 0.0
		self.value: float = 0.0

		self.register_property(
			name='normalValue',
			class_type=float,
			multiplier=UnitMultiplier.k,
			unit=UnitSymbol.V,
			description='''Electrical voltage, can be both AC and DC.''',
			profiles=[]
		)
		self.register_property(
			name='value',
			class_type=float,
			multiplier=UnitMultiplier.k,
			unit=UnitSymbol.V,
			description='''Electrical voltage, can be both AC and DC.''',
			profiles=[]
		)
