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
from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.identified_object import IdentifiedObject
from GridCalEngine.IO.cim.cgmes.cgmes_enums import cgmesProfile, SynchronousMachineKind, GeneratorControlSource, WindingConnection, ControlAreaTypeKind, SVCControlMode, BatteryStateKind, DCPolarityKind, FuelType, AsynchronousMachineKind, PhaseCode, Currency, HydroPlantStorageKind, HydroTurbineKind, CurveStyle, OperationalLimitDirectionKind, LimitKind, UnitSymbol, CsPpccControlKind, CsOperatingModeKind, HydroEnergyConversionKind, SynchronousMachineOperatingMode, DCConverterOperatingModeKind, UnitMultiplier, RegulatingControlModeKind


class TransformerEnd(IdentifiedObject):
	def __init__(self, rdfid='', tpe='TransformerEnd'):
		IdentifiedObject.__init__(self, rdfid, tpe)

		from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.base_voltage import BaseVoltage
		self.BaseVoltage: BaseVoltage | None = None
		from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.phase_tap_changer import PhaseTapChanger
		self.PhaseTapChanger: PhaseTapChanger | None = None
		from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.ratio_tap_changer import RatioTapChanger
		self.RatioTapChanger: RatioTapChanger | None = None
		from GridCalEngine.IO.cim.cgmes.cgmes_v3_0_0.devices.terminal import Terminal
		self.Terminal: Terminal | None = None
		self.endNumber: int = 0

		self.register_property(
			name='BaseVoltage',
			class_type=BaseVoltage,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''Base voltage of the transformer end.  This is essential for PU calculation.''',
			profiles=[]
		)
		self.register_property(
			name='PhaseTapChanger',
			class_type=PhaseTapChanger,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''Phase tap changer associated with this transformer end.''',
			profiles=[]
		)
		self.register_property(
			name='RatioTapChanger',
			class_type=RatioTapChanger,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''Ratio tap changer associated with this transformer end.''',
			profiles=[]
		)
		self.register_property(
			name='Terminal',
			class_type=Terminal,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''Terminal of the power transformer to which this transformer end belongs.''',
			profiles=[]
		)
		self.register_property(
			name='endNumber',
			class_type=int,
			multiplier=UnitMultiplier.none,
			unit=UnitSymbol.none,
			description='''Number for this transformer end, corresponding to the end's order in the power transformer vector group or phase angle clock number.  Highest voltage winding should be 1.  Each end within a power transformer should have a unique subsequent end number.   Note the transformer end number need not match the terminal sequence number.''',
			profiles=[]
		)
