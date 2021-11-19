# Implementation of an interferometric direction finding algorithm for an orthoganal L antenna array
#
#
# 90 degrees
# |
# v
#
# 7
# |
# |
# |
# |
# 5
# |
# |
# 3
# |
# 1--2-----4------------6		<-- 0 degrees
#
# TODO: Fix ambiguities when the wavefront is parallel to one of the array's legs.
#       In this situation the aforementioned leg provides no information (phase deltas are zero).
#       So the linear regression tends to fit two DOAs +-180 degrees apart equally well.
#

from numpy import array, arange, iinfo, int32
from numpy.linalg import inv, norm
from math import pi, cos, sin, ceil
from cmath import phase, rect
from functools import reduce
import matplotlib.pyplot as plt

def create_bases_mat(base_sizes):
    num_bases = len(base_sizes)
    bases = arange(num_bases * 4).reshape((num_bases * 2, 2))

    for i, base_size in enumerate(base_sizes[::-1]):
        bases[i][0] = 0
        bases[i][1] = base_size

    for i, base_size in enumerate(base_sizes):
        bases[i + num_bases][0] = base_size
        bases[i + num_bases][1] = 0

    return bases

def calculate_df(bases, phases) -> complex:
    bases_t = bases.T
    A = bases_t.dot(bases)
    B = inv(A)
    C = B.dot(bases_t)

    df = C.dot(phases)

    x = df[1][0]
    y = df[0][0]

    return x + y*1j

def calculate_azimuth_deg(df):
    return phase(df) * 180 / pi

def simulate_phases(bases, azimuth_deg):
    simulated_azimuth_rad = azimuth_deg * pi / 180
    simulated_df = rect(1, simulated_azimuth_rad)
    simulated_df_mat = array([
        [simulated_df.imag],
        [simulated_df.real],
    ])
    return bases.dot(simulated_df_mat)

def calculate_phases(apertures):
    phases = [phase(ap) for ap in apertures]
    phases = phases[0::2] + phases[1::2]
    return phases

def calculate_apertures(base_sizes, azimuth_deg, frequency_Hz):
    c = 299_792_458
    azimuth_rad = azimuth_deg * pi / 180
    apertures = []
    for base_size in base_sizes:
        max_phase_delta = 2 * pi * base_size * frequency_Hz / c
        x_base = max_phase_delta * cos(azimuth_rad)
        y_base = max_phase_delta * sin(azimuth_rad)
        apertures.append(rect(1, x_base))
        apertures.append(rect(1, y_base))
    return apertures

def get_OLS_linear_regression_error(x, y):
    s_x = sum(x)
    s_y = sum(y)
    s_xx = reduce(lambda sum, val: sum + val*val, x, 0)
    s_xy = reduce(lambda sum, x_y: (sum[0] + x_y[0] * x_y[1], 0), zip(x, y), (0,0))[0]
    n = len(x)

    beta = (n * s_xy - s_x * s_y) / (n * s_xx - s_x*s_x)
    alpha = (s_y - beta * s_x) / n

    g = [beta * x_val + alpha for x_val in x]

    error = reduce(lambda sum, y_g: (sum[0] + pow(y_g[0] - y_g[1], 2), 0), zip(y, g), (0,0))[0]

    return error

def disambiguate_phases(phases, base_sizes, frequency_Hz):
    phases = [phase * 180 / pi for phase in phases]

    c = 299_792_458
    max_phase = ceil(360 * base_sizes[0] * frequency_Hz / c)
    min_phase = -max_phase

    first_base_phase = phases[0]
    phase_sets = [
        [first_base_phase]
    ]
    phase = first_base_phase + 360
    while phase <= max_phase:
        phase_sets.append([phase])
        phase += 360
    phase = first_base_phase - 360
    while phase >= min_phase:
        phase_sets.append([phase])
        phase -= 360

    for phase_set in phase_sets:
        for phase_index in range(1, len(phases)):
            extrapolated_phase = phase_set[phase_index - 1] * 2.5
            delta = phases[phase_index] - extrapolated_phase
            ambiguity = 360 * round(delta / 360)
            phase_set.append(phases[phase_index] - ambiguity)

    # num_ambiguities = len(phase_sets)
    # print(num_ambiguities)

    unambiguous_phases = []
    min_error = iinfo(int32).max
    for phase_set in phase_sets:
        x = [0] + base_sizes
        y = [0] + phase_set
        error = get_OLS_linear_regression_error(x, y)
        if error < min_error:
            min_error = error
            unambiguous_phases = phase_set

    unambiguous_phases = [phase * pi / 180 for phase in unambiguous_phases]

    return unambiguous_phases

def normalise(v):
    k = norm(v)
    if k == 0:
        return v
    return v / k

def do_df_algo(frequency_Hz, simulated_azimuth_deg, base_sizes):
    bases = create_bases_mat(base_sizes)

    apertures = calculate_apertures(base_sizes, simulated_azimuth_deg, frequency_Hz)

    # phases = simulate_phases(bases, simulated_azimuth_deg)

    phases = calculate_phases(apertures)
    phases[:3:] = disambiguate_phases(phases[:3:], base_sizes, frequency_Hz)[::-1]
    phases[3::] = disambiguate_phases(phases[3::], base_sizes, frequency_Hz)
    phases = array(phases).reshape(6,1)

    df = calculate_df(bases, phases)

    # simulated_phases = simulate_phases(bases, 2)
    # simulated_phases = normalise(simulated_phases)
    # print(simulated_phases)
    # phases = normalise(phases)
    # print(phases)

    azimuth = round(calculate_azimuth_deg(df))

    return azimuth

frequency_Hz = 30_000_000
base_sizes = [13.6, 34, 85]

x = range(-180, 180)
y = x
g = []
for simulated_azimuth_deg in x:
    az = do_df_algo(frequency_Hz, simulated_azimuth_deg, base_sizes)
    if(simulated_azimuth_deg != az):
        print(f"expected: {simulated_azimuth_deg}, got: {az}")
    g.append(az)

plt.plot(x,y,x,g)
plt.show()
