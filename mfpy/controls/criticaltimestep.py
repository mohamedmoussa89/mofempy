"""
Calculate critical timestep
"""

from numpy import sqrt
def calculate_nodal_critical_timestep(M, K_sum):
    """
    R.F.  Kulak,
    “Critical  Time  Step  Estimation  for  Three-Dimensional  Explicit  Impact Analysis”,
    Structures under Shock and Impact – Proceedings of the First International Conference (1989)
    """
    return min(2*sqrt(M/K_sum))





