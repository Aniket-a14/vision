"""Physical and alloy constants for Al-Si high-pressure die casting."""

from typing import Final

# Hydrogen solubility in aluminium at 660 C, mL H2 per 100 g.
H2_SOLUBILITY_LIQUID: Final = 0.69
H2_SOLUBILITY_SOLID: Final = 0.036
H2_REFERENCE_TEMP_C: Final = 660.0

# Van 't Hoff slope fitted so solubility rises ~0.69 -> ~0.92 mL/100g over 660-700 C.
H2_DISSOLUTION_SLOPE: Final = 6480.0

# Taylor's relation for the critical iron content: Fe_crit = SLOPE * Si + INTERCEPT.
FE_CRIT_SI_SLOPE: Final = 0.075
FE_CRIT_INTERCEPT: Final = -0.05

# Below this iron level the melt attacks H13 die steel and solders.
FE_SOLDERING_FLOOR_PCT: Final = 0.70

# Sludge factor SF = Fe + 2*Mn + 3*Cr must stay under this at holding temperature.
SLUDGE_FACTOR_LIMIT: Final = 1.8
SLUDGE_MN_WEIGHT: Final = 2.0
SLUDGE_CR_WEIGHT: Final = 3.0

# Intensification pressure stops paying back above this knee, in MPa.
PRESSURE_SATURATION_MPA: Final = 67.4

# Die surface temperature window for a thin-wall impeller, in C.
DIE_TEMP_MIN_C: Final = 180.0
DIE_TEMP_MAX_C: Final = 280.0

# A local cold spot this far below the die average can trigger a cold shut.
DIE_COLD_SPOT_DELTA_C: Final = 30.0

# Melt loses this much between runner and gate, in C.
RUNNER_TO_GATE_LOSS_C: Final = 85.0

# Solidification onset for A380/ADC12, in C.
LIQUIDUS_C: Final = 595.0
SOLIDUS_C: Final = 540.0
