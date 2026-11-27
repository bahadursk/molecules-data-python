"""chemtools: a few handy chemistry functions."""
R = 8.314462618  # J/(mol K)

def kelvin(t_c):
    return t_c + 273.15

def ideal_gas_volume(n, T, P=101325):
    return n * R * T / P
