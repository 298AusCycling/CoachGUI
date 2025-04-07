# app.py

import streamlit as st
import pandas as pd
import time
from iteration import simulate_race

st.title("Team Pursuit Race Simulator")

# A place to store our “database” of past simulations
if "simulation_history" not in st.session_state:
    st.session_state["simulation_history"] = []

drafting_percents = [1.0, 0.58, 0.52, 0.53]

def switch_schedule_description(switch_schedule):
    laps = [i+1 for i, v in enumerate(switch_schedule) if v == 1]
    return laps

uploaded_file = st.file_uploader("Upload Performance Data Excel File", type=["xlsx"])

if uploaded_file:
    df_athletes = pd.read_excel(uploaded_file)
    power_duration_df = pd.read_excel(uploaded_file, sheet_name="Power Curves")

    available_athletes = (
        df_athletes["Name"]
        .str.extract(r'M(\d+)')[0]
        .dropna()
        .astype(int)
        .tolist()
    )
    chosen_athletes = st.segmented_control("Select 4 Athletes", options=available_athletes, selection_mode = 'multi', default=None, key = 1323)
    st.markdown(f"Selected Riders: {sorted(chosen_athletes)}.")
    if len(chosen_athletes) == 4:
        start_order = st.segmented_control("Initial Rider Order", options=sorted(chosen_athletes), selection_mode = 'multi', default=None, key = 1231)
        st.markdown(f"Initial Starting Order: {start_order}")

        st.subheader("Switch Schedule (32 half-laps)")
        switch_schedule = []
        peel_schedule = []

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Switch (1 = switch after this half-lap)**")
            for i in range(32):
                val = st.checkbox(f"{i+1}", key=f"switch_{i}")
                switch_schedule.append(1 if val else 0)

        with col2:
            st.markdown("**Peel (1 = peel here)**")
            for i in range(32):
                val = st.checkbox(f"{i+1}", key=f"peel_{i}")
                peel_schedule.append(1 if val else 0)

        # Find the first peel
        try:
            peel_location = peel_schedule.index(1)
        except ValueError:
            peel_location = None

        if peel_location is None:
            st.warning("Please select at least one peel location.")
        else:
            if st.button("Simulate Race"):
                with st.spinner("Running simulation..."):
                    final_order, final_time, final_distance, final_half_lap_count, W_rem = simulate_race(
                        switch_schedule=switch_schedule,
                        chosen_athletes=chosen_athletes,
                        start_order=start_order,
                        drafting_percents=drafting_percents,
                        peel_location=peel_location + 1,  # Adjust if needed
                        power_duration_df=power_duration_df,
                        df_athletes=df_athletes,
                        total_mass=70,
                        v0=0.5,
                        rho=1.225
                    )

                st.success("✅ Simulation Complete!")
                st.write(f"**Final Order:** {final_order}")
                st.write(f"**Total Time:** {final_time:.2f} seconds")
                st.write(f"**Total Distance:** {final_distance:.2f} m")
                st.write(f"**Half Laps Completed:** {final_half_lap_count}")
                st.write(f"**Switch at half-laps:** {switch_schedule_description(switch_schedule)}")

                st.subheader("W′ Remaining per Rider:")
                for k, v in W_rem.items():
                    st.write(f"{k}: {v:.1f} J")

                # ------------------------------
                # Save this run to our "database"
                # ------------------------------
                simulation_record = {
                    "timestamp": time.time(),
                    "chosen_athletes": chosen_athletes,
                    "start_order": start_order,
                    "switch_schedule": switch_schedule,
                    "peel_location": peel_location,
                    "final_order": final_order,
                    "final_time": final_time,
                    "final_distance": final_distance,
                    "final_half_lap_count": final_half_lap_count,
                    "W_rem": W_rem
                }
                st.session_state["simulation_history"].append(simulation_record)
                st.info("Simulation saved in history!")

# ------------------------------
# 2) Display & manage all past simulations
# ------------------------------
st.header("Past Simulations")

# If there's something in simulation_history, display them
if len(st.session_state["simulation_history"]) == 0:
    st.write("No simulations saved yet.")
else:
    # We can display each simulation as a collapsible expander,
    # or as a table or something else. Here’s an example:
    for i, sim in enumerate(st.session_state["simulation_history"]):
        with st.expander(f"Simulation {i+1} (timestamp: {sim['timestamp']})"):
            st.write(f"**Riders:** {sim['chosen_athletes']}")
            st.write(f"**Start Order:** {sim['start_order']}")
            st.write(f"**Switch Schedule:** {sim['switch_schedule']}")
            st.write(f"**Peel Location (0-based):** {sim['peel_location']}")
            st.write(f"**Final Order:** {sim['final_order']}")
            st.write(f"**Final Time:** {sim['final_time']:.2f} s")
            st.write(f"**Final Distance:** {sim['final_distance']:.2f} m")
            st.write(f"**Final Half Laps:** {sim['final_half_lap_count']}")
            st.write("**W′ Remaining:**")
            for k, v in sim['W_rem'].items():
                st.write(f"{k}: {v:.1f} J")

            # 3) Delete button for each simulation
            delete_button_key = f"delete_sim_{i}"
            if st.button(f"Delete Simulation {i+1}", key=delete_button_key):
                st.session_state["simulation_history"].pop(i)
                st.experimental_rerun()
