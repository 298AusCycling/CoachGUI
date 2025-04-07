# %% [markdown]
# **combining acc phase and the SS phase**
# 
# - currently doesn't have valid combinations because the W' is depleated (but it does track this)

# %% [markdown]
# imports


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import root_scalar
import scipy
from scipy.integrate import solve_ivp
# from scipy.integrate import cumtrapz  
from scipy.interpolate import interp1d
import scipy
from scipy.integrate import solve_ivp
# from scipy.integrate import cumtrapz  
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

# %% [markdown]
# reading in the data from the sheet

# %%
#reading in data from spreadsheet
df = pd.read_excel('2024_Performance_Physiology_Data_Anonymised_v2.xlsx')

# Load the data
power_duration_df = pd.read_excel("2024_Performance_Physiology_Data_Anonymised_v2.xlsx", sheet_name="Power Curves")

# %% [markdown]
# general helper functions

# %%
rho = 1.225 #kg/m^3
def get_rider_info(num):
    #athlete number
    athlete_name = f'M{num}'
    athlete = df[df['Name'] == athlete_name]

    #computing their critical power curve
    W_prime = athlete["W'"].iloc[0]
    W_prime = W_prime * 1000 #convert to J
    t_prime = athlete['Tmax'].iloc[0]
    CP = athlete['CP'].iloc[0]
    AC = athlete['CdA'].iloc[0]
    Pmax = athlete['Pmax'].iloc[0]

    return W_prime, t_prime, CP, AC, Pmax

def power_from_velo(velo, AC, rho = 1.225):
    return 0.5*rho*AC*velo**3

def velocity(P, AC, rho = 1.225):
    return (2*P/(rho*AC))**(1/3)

def power_on_crit_curve(W_prime,t, CP, Pmax):
    P = W_prime/t + CP
    if P > Pmax:
        P = Pmax
    return P

def rotate_list(lst):
    return lst[1:] + [lst[0]]  # Moves first element to the back


# %% [markdown]
# **new helper functions**

# %% [markdown]
# for acceleration phase

# %%
def cumulative_trapezoid(y, x):
    result = np.zeros_like(y)
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2
    result[1:] = np.cumsum(avg_y * dx)
    return result

def make_v_of_t_numerical(power, Mtot, rho, AC, v0):
    def dvdt(t, v):
        v = v[0]
        if v <= 0:
            return [0]
        drag = 0.5 * rho * AC * v**3
        return [(power - drag) / (Mtot * v)]

    t_span = (0, 100)
    t_eval = np.linspace(*t_span, 1000)
    sol = solve_ivp(dvdt, t_span, [v0], t_eval=t_eval, rtol=1e-8, atol=1e-10)

    t_vals = sol.t
    v_vals = sol.y[0]

    v_func = interp1d(t_vals, v_vals, kind='cubic', fill_value='extrapolate')
    l_vals = cumulative_trapezoid(v_vals, t_vals)
    l_func = interp1d(t_vals, l_vals, kind='cubic', fill_value='extrapolate')

    return v_func, l_func

# %%
def power_to_time(power, Mtot, rho, AC, num_laps_till_switch, v0, t0):
    l_target = 125 * num_laps_till_switch
    v_func, l_func = make_v_of_t_numerical(power, Mtot, rho, AC, v0)

    def objective(t):
        return l_func(t) - l_target

    result = root_scalar(objective, bracket=[1, 100], method='brentq')
    
    return result.root if result.converged else np.nan

# %%
def find_self_consistent_power(initial_power_guess, power_duration_df, rider_col, Mtot, rho, AC, num_laps_till_switch, v0, epsilon=1.0, max_iter=50):
    # Extract time and power for the rider
    times = power_duration_df["Time (s)"].values
    rider_power = power_duration_df[rider_col].values
    
    # Create interpolator for power-duration curve
    power_curve = interp1d(times, rider_power, kind='linear', fill_value='extrapolate')

    power_guess = initial_power_guess
    for i in range(max_iter):
        # Step 1: get time from physics model
        t_result = power_to_time(power_guess, Mtot, rho, AC, num_laps_till_switch, v0, t0=0)

        if np.isnan(t_result) or t_result <= 0:
            print(f"[WARNING] Iteration {i}: invalid time returned for power = {power_guess:.2f}")
            return np.nan, np.nan

        # Step 2: get power from physiology model
        power_actual = power_curve(t_result)

        # Step 3: check convergence
        diff = abs(power_actual - power_guess)
        print(f"Iteration {i}: power_guess = {power_guess:.2f}, time = {t_result:.2f}, power_actual = {power_actual:.2f}, diff = {diff:.2f}")

        if diff < epsilon:
            return power_actual, t_result

        # Step 4: update guess
        power_guess = power_actual

    print("[WARNING] Maximum iterations reached without convergence.")
    return power_guess, t_result

# %% [markdown]
# for the SS phase

# %%
def get_power_duration_curve(power_duration_df, rider_id):
    times = power_duration_df["Time (s)"].values
    rider_power = power_duration_df[rider_id].values

    # Create interpolator for power-duration curve
    power_curve = interp1d(times, rider_power, kind='linear', fill_value='extrapolate')
    return power_curve

# %%
def get_half_lap_velo(power_duration_df, switch_schedule, current_order, chosen_athletes, half_lap_completed, rho = 1.225):
    rider_data = {}
    for i in range(4):
        athlete_num = chosen_athletes[i]
        W_prime, t_prime, CP, AC, Pmax = get_rider_info(athlete_num)

        # Store in dictionary with formatted variable names
        rider_data[f"W{athlete_num}_prime"] = W_prime
        rider_data[f"t{athlete_num}_prime"] = t_prime
        rider_data[f"CP{athlete_num}"] = CP
        rider_data[f"AC{athlete_num}"] = AC
        rider_data[f"Pmax{athlete_num}"] = Pmax

    lead_rider = current_order[0]
    lead_rider_id = f'M{lead_rider}'

    power_curve = get_power_duration_curve(power_duration_df, lead_rider_id)

    rider_data[f"CP{lead_rider}"]
    AC = rider_data[f"AC{lead_rider}"]
    rest_switch_schedule = switch_schedule[half_lap_completed-1:]
    half_lap_dist = 125  # in meters
    dist_till_switch = rest_switch_schedule.index(1) * half_lap_dist
    print("Distance till switch:", dist_till_switch)

    # Define the function to find the root of
    def equation_to_solve(t):
        if t <= 0:
            return np.inf  # Prevent divide by zero or negative time
        aerodynamic_power = 0.5 * rho * AC * (dist_till_switch / t)**3
        return aerodynamic_power - power_curve(t)
    
    # Plot setup -- seeing if they intersect
    print(f"AC (CdA) for rider {lead_rider_id}: {AC}")
    t_vals = np.linspace(1, 1000, 500)
    aero_power_vals = [0.5 * rho * AC * (dist_till_switch / t)**3 for t in t_vals]
    cp_power_vals = power_curve(t_vals)

    plt.figure(figsize=(10, 5))
    plt.plot(t_vals, aero_power_vals, label='Aerodynamic Power Required', linewidth=2)
    plt.plot(t_vals, cp_power_vals, label='Rider Critical Power Curve', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title(f"Power vs Time for Rider {lead_rider_id}")
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--')

    # Use root_scalar to solve the equation
    # result = root_scalar(equation_to_solve, bracket=[1, 1000], method='brentq')
    # if result.converged:
    #     tau =  result.root
    # else:
    #     print("Solver did not converge.")
    try:
        result = root_scalar(equation_to_solve, bracket=[1, 1000], method='brentq')
        if result.converged:
            tau = result.root
        else:
            raise ValueError("Root-finding did not converge.")
    except ValueError as e:
        print("Root finding failed:", e)
        print("f(1) =", equation_to_solve(1))
        print("f(1000) =", equation_to_solve(1000))
        return None  # or some fallback

    velo = dist_till_switch / tau  # Velocity in m/s
    return velo

# %%
def get_chunk_velocity(power_duration_df, lead_rider_id, AC, dist, rho=1.225):
    """
    Solve for velocity over a chunk given the lead rider and distance.

    Args:
        power_duration_df: DataFrame with time and rider power data
        lead_rider_id: string like 'M1'
        AC: rider's CdA (frontal area * drag coefficient)
        dist: total distance for this chunk (in meters)
        rho: air density (default 1.225)

    Returns:
        Velocity (m/s) over the chunk
    """

    power_curve = get_power_duration_curve(power_duration_df, lead_rider_id)

    def equation_to_solve(t):
        if t <= 0:
            return np.inf
        aerodynamic_power = 0.5 * rho * AC * (dist / t)**3
        return aerodynamic_power - power_curve(t)

    # Try a smart bracket for root-finding
    try:
        result = root_scalar(equation_to_solve, bracket=[1, 1000], method='brentq')
        if result.converged:
            tau = result.root
            return dist / tau  # velocity = distance / time
        else:
            raise ValueError("Root-finding did not converge.")
    except Exception as e:
        print(f"⚠️ Failed to solve for chunk velocity: {e}")
        print(f"f(1) = {equation_to_solve(1)}, f(1000) = {equation_to_solve(1000)}")
        return None


# %%
def get_chunks_from_schedule(switch_schedule, half_lap_completed, peel_location=None, half_lap_dist=125):
    chunks = []
    current_chunk = []
    chunk_start_idx = half_lap_completed - 1  # where we start processing

    rest_schedule = switch_schedule[chunk_start_idx:]
    total_half_laps = len(rest_schedule)

    for i, val in enumerate(rest_schedule):
        current_chunk.append(val)

        # Check if it's a switch
        if val == 1:
            chunk_half_laps = len(current_chunk)
            switch_idx = chunk_start_idx + chunk_half_laps  # 1-based index (half-lap after this one)

            # Break early if peel_location exceeded
            if peel_location is not None and switch_idx > peel_location:
                break

            chunk_half_lap_indices = list(range(chunk_start_idx, chunk_start_idx + chunk_half_laps))
            distance = sum(127 if switch_schedule[i] == 1 else 125 for i in chunk_half_lap_indices)

            chunks.append({
                'start_idx': chunk_start_idx,
                'num_half_laps': chunk_half_laps,
                'distance': distance,
                'switch_idx': switch_idx
            })

            current_chunk = []
            chunk_start_idx += chunk_half_laps

    # Handle trailing chunk (if it doesn’t end in a 1)
    if current_chunk:
        chunk_half_laps = len(current_chunk)
        chunk_half_lap_indices = list(range(chunk_start_idx, chunk_start_idx + chunk_half_laps))
        distance = sum(127 if switch_schedule[i] == 1 else 125 for i in chunk_half_lap_indices)

        chunks.append({
            'start_idx': chunk_start_idx,
            'num_half_laps': chunk_half_laps,
            'distance': distance,
            'switch_idx': None  # no switch at the end
        })

    return chunks


# %% [markdown]
# **now we have the functions for the 3 phases**

# %%
def accel_phase(v0, total_mass, switch_schedule, chosen_athletes, start_order, drafting_percents, power_duration_df, rho=1.225):
    epsilon = 0.5
    # Find the first switch in the schedule
    try:
        next_switch_idx = switch_schedule.index(1)
    except ValueError:
        raise ValueError("Switch schedule has no switches — cannot determine end of acceleration phase.")

    # Calculate how many half-laps the acceleration phase covers
    num_laps_till_switch = next_switch_idx + 1  # since index 0 means 1st half-lap
    total_distance = 125 * num_laps_till_switch
    current_order = start_order.copy()
    
    rider_data = {}
    W_rem = {}

    # Build rider data and initial W'
    for rider in chosen_athletes:
        W_prime, t_prime, CP, AC, Pmax = get_rider_info(rider)
        rider_data[rider] = {
            "W_prime": W_prime,
            "t_prime": t_prime,
            "CP": CP,
            "AC": AC,
            "Pmax": Pmax,
        }
        W_rem[f"W{rider}_rem"] = W_prime

    # Solve for lead rider power + time
    lead_rider = current_order[0]
    AC_lead = rider_data[lead_rider]["AC"]
    rider_col = f"M{lead_rider}"

    initial_guess = 700  # or a rough estimate
    final_power, final_time = find_self_consistent_power(
        initial_power_guess=initial_guess,
        power_duration_df=power_duration_df,
        rider_col=rider_col,
        Mtot=total_mass,
        rho=rho,
        AC=AC_lead,
        num_laps_till_switch=num_laps_till_switch,
        v0=v0,
        epsilon=epsilon
    )

    # Update W' for each rider during the acceleration phase
    for i, rider in enumerate(current_order):
        CP = rider_data[rider]["CP"]
        draft = drafting_percents[i]
        power_curve = get_power_duration_curve(power_duration_df, f"M{rider}")
        rider_power = power_curve(final_time) * draft
        delta_W = (rider_power - CP) * final_time
        W_rem[f"W{rider}_rem"] -= delta_W
        if W_rem[f"W{rider}_rem"] < 0:
            print(f"⚠️ Rider {rider} exceeded W′ during acceleration!")

    half_lap_completed = num_laps_till_switch  # e.g., 3

    return current_order, final_time, total_distance, half_lap_completed, W_rem


# %%
def SS_4rider(power_duration_df, switch_schedule, peel_location, current_order, chosen_athletes, time, distance, W_rem, half_lap_completed, drafting_percents):
    # Get rider info
    rider_data = {}
    for i in range(4):
        athlete_num = chosen_athletes[i]
        W_prime, t_prime, CP, AC, Pmax = get_rider_info(athlete_num)
        rider_data[f"W{athlete_num}_prime"] = W_prime
        rider_data[f"t{athlete_num}_prime"] = t_prime
        rider_data[f"CP{athlete_num}"] = CP
        rider_data[f"AC{athlete_num}"] = AC
        rider_data[f"Pmax{athlete_num}"] = Pmax

    total_time = time
    total_distance = distance
    order = current_order
    count = half_lap_completed
    # rest_switch_schedule = switch_schedule[half_lap_completed - 1:]
    # print("Rest switch schedule:", rest_switch_schedule)

    # Get race chunks between now and peel
    chunks = get_chunks_from_schedule(switch_schedule, half_lap_completed, peel_location)
    print(f"Chunks: {chunks}")

    for chunk in chunks:
        print(chunk)
        chunk_distance = chunk['distance']
        switch_idx = chunk['switch_idx']

        lead_rider = order[0]
        lead_rider_id = f'M{lead_rider}'
        AC = rider_data[f"AC{lead_rider}"]

        # Solve for velocity
        velo = get_chunk_velocity(power_duration_df, lead_rider_id, AC, chunk_distance)
        # count = chunk['switch_idx'] if chunk['switch_idx'] is not None else (chunk['start_idx'] + chunk['num_half_laps'])
        if velo is None:
            print(f"Invalid velocity at chunk ending in switch {switch_idx}")
            return order, total_time, total_distance, count, W_rem

        chunk_time = chunk_distance / velo
        start_hlap = chunk['start_idx'] + 1
        end_hlap = chunk['start_idx'] + chunk['num_half_laps']
        print(f"Chunk from half-lap {start_hlap} to {end_hlap}")

        # Update W′ for each rider
        for i in range(4):
            rider_id = order[i]
            cp = rider_data[f"CP{rider_id}"]
            power = get_power_duration_curve(power_duration_df, f'M{rider_id}')(chunk_time) * drafting_percents[i]
            W_rem[f"W{rider_id}_rem"] -= (power - cp) * chunk_time
            if W_rem[f"W{rider_id}_rem"] < 0:
                print(f"⚠️ Rider {rider_id} over threshold! W′ exhausted.")

        # Update time, distance, lap count
        total_time += chunk_time
        total_distance += chunk_distance
        count += chunk['num_half_laps']

        # Update order after switch
        order = rotate_list(order)

        # # Stop if we've reached the peel location
        # if switch_idx == peel_location:
        #     print(f"Peeling at switch {switch_idx}")
        #     order = order[1:]  # Remove lead rider
        #     break
    print(f"Peeling at switch {switch_idx}")
    order = order[1:]  # Remove lead rider
    return order, total_time, total_distance, count, W_rem


# %%
def SS_3rider(power_duration_df, switch_schedule, current_order, chosen_athletes, time, distance, W_rem, half_lap_completed, drafting_percents):
    # Get rider info
    rider_data = {}
    for i in range(4):
        athlete_num = chosen_athletes[i]
        W_prime, t_prime, CP, AC, Pmax = get_rider_info(athlete_num)
        rider_data[f"W{athlete_num}_prime"] = W_prime
        rider_data[f"t{athlete_num}_prime"] = t_prime
        rider_data[f"CP{athlete_num}"] = CP
        rider_data[f"AC{athlete_num}"] = AC
        rider_data[f"Pmax{athlete_num}"] = Pmax

    total_time = time
    total_distance = distance
    order = current_order  # Should now have only 3 riders
    print(f"Current order: {order}")
    count = half_lap_completed
    # rest_switch_schedule = switch_schedule[half_lap_completed-1:]
    # print("Rest switch schedule:", rest_switch_schedule)

    chunks = get_chunks_from_schedule(switch_schedule, half_lap_completed)
    print(f"Chunks: {chunks}")

    for chunk in chunks:
        # print(chunk)
        chunk_distance = chunk['distance']
        switch_idx = chunk['switch_idx']

        lead_rider = order[0]
        lead_rider_id = f'M{lead_rider}'
        AC = rider_data[f"AC{lead_rider}"]

        # Solve for velocity
        velo = get_chunk_velocity(power_duration_df, lead_rider_id, AC, chunk_distance)
        # count = chunk['switch_idx'] if chunk['switch_idx'] is not None else (chunk['start_idx'] + chunk['num_half_laps'])
        if velo is None:
            print(f"Invalid velocity at chunk ending in switch {switch_idx}")
            return order, total_time, total_distance, count, W_rem

        chunk_time = chunk_distance / velo
        start_hlap = chunk['start_idx'] + 1
        end_hlap = chunk['start_idx'] + chunk['num_half_laps']
        print(f"Chunk from half-lap {start_hlap} to {end_hlap}")

        # Update W′ for each rider (just top 3 now)
        for i in range(3):
            rider_id = order[i]
            cp = rider_data[f"CP{rider_id}"]
            power = get_power_duration_curve(power_duration_df, f'M{rider_id}')(chunk_time) * drafting_percents[i]
            W_rem[f"W{rider_id}_rem"] -= (power - cp) * chunk_time
            if W_rem[f"W{rider_id}_rem"] < 0:
                print(f"⚠️ Rider {rider_id} over threshold! W′ exhausted.")

        # Update time, distance, lap count
        total_time += chunk_time
        total_distance += chunk_distance
        count += chunk['num_half_laps']

        # Update order after switch
        order = rotate_list(order)

    return order, total_time, total_distance, count, W_rem


# %% [markdown]
# **okay now we use them**

# %%
def simulate_race(
    switch_schedule,
    chosen_athletes,
    start_order,
    drafting_percents,
    peel_location,
    power_duration_df,
    total_mass=70,
    v0=0.5,
    rho=1.225
):
    """
    Simulates the full team pursuit race with acceleration, 4-rider steady-state, and 3-rider steady-state phases.

    Returns:
        final_order, final_time, final_distance, final_half_lap_count, W_rem
    """
    print("\n=== Acceleration Phase ===")
    order, total_time, total_distance, half_lap_completed, W_rem = accel_phase(
        v0=v0,
        total_mass=total_mass,
        switch_schedule=switch_schedule,
        chosen_athletes=chosen_athletes,
        start_order=start_order,
        drafting_percents=drafting_percents,
        power_duration_df=power_duration_df,
        rho=rho
    )

    print("\n=== 4-Rider Steady-State Phase ===")
    order, total_time, total_distance, half_lap_completed, W_rem = SS_4rider(
        power_duration_df=power_duration_df,
        switch_schedule=switch_schedule,
        peel_location=peel_location,
        current_order=order,
        chosen_athletes=chosen_athletes,
        time=total_time,
        distance=total_distance,
        W_rem=W_rem,
        half_lap_completed=half_lap_completed,
        drafting_percents=drafting_percents
    )

    next_half_lap_completed = half_lap_completed + 1

    print("\n=== 3-Rider Steady-State Phase ===")
    final_order, final_time, final_distance, final_half_lap_count, W_rem = SS_3rider(
        power_duration_df=power_duration_df,
        switch_schedule=switch_schedule,
        current_order=order,
        chosen_athletes=chosen_athletes,
        time=total_time,
        distance=total_distance,
        W_rem=W_rem,
        half_lap_completed=next_half_lap_completed,
        drafting_percents=drafting_percents
    )

    return final_order, final_time, final_distance, final_half_lap_count, W_rem


# %%
# Define inputs
switch_schedule = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                   0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]  # length = 32

chosen_athletes = [1, 2, 3, 4]  # athlete numbers as in spreadsheet
start_order = [1, 2, 3, 4]      # initial order of riders
drafting_percents = [1.0, 0.58, 0.52, 0.53]  # estimated drag multipliers
peel_location = 24  # switch index where one rider peels off

# Run full race simulation
final_order, final_time, final_distance, final_half_lap_count, W_rem = simulate_race(
    switch_schedule=switch_schedule,
    chosen_athletes=chosen_athletes,
    start_order=start_order,
    drafting_percents=drafting_percents,
    peel_location=peel_location,
    power_duration_df=power_duration_df
)

# Output results
print("\n=== Race Summary ===")
print(f"Finish order: {final_order}")
print(f"Total time: {final_time:.2f} s")
# print(f"Total distance: {final_distance:.2f} m")
# print(f"Total half-laps: {final_half_lap_count}")
for k, v in W_rem.items():
    print(f"{k} remaining: {v:.1f} J")

print(switch_schedule)

