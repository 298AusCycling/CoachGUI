# app.py

import streamlit as st
import pandas as pd
from iteration import simulate_race

st.title("Team Pursuit Race Simulator")

drafting_percents = [1.0, 0.58, 0.52, 0.53]

def switch_schedule_description(switch_schedule):
    """
    Helper for displaying which half-laps had switches.
    """
    laps = []
    for i, val in enumerate(switch_schedule):
        if val == 1:
            # half-laps are 1-indexed in user display
            laps.append(i + 1)
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

        st.subheader("Define Switch Schedule (32 half-laps)")

        switch_schedule = []
        peel_schedule = []


        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Switch (✅ = switch after this half-lap)**")
            for i in range(32):
                val = st.checkbox(f"{i+1}", key=f"switch_{i}")
                switch_schedule.append(1 if val else 0)

        with col2:
            st.markdown("**Peel (✅ = peel here)**")
            for i in range(32):
                val = st.checkbox(f"{i+1}", key=f"peel_{i}")
                peel_schedule.append(1 if val else 0)

        try:
            peel_location = peel_schedule.index(1)  # 0-based index
        except ValueError:
            peel_location = None
        if peel_location is None:
            st.warning("Please select at least one peel location.")
        else:
            if st.button("Simulate Race"):
                with st.spinner("Running simulation..."):
                    # 4) Call simulate_race from iteration.py
                    final_order, final_time, final_distance, final_half_lap_count, W_rem = simulate_race(
                        switch_schedule=switch_schedule,
                        chosen_athletes=chosen_athletes,
                        start_order=start_order,
                        drafting_percents=drafting_percents,
                        peel_location=peel_location + 1,  
                        power_duration_df=power_duration_df,
                        df_athletes=df_athletes,
                    )

                # 5) Display results
                st.success("✅ Simulation Complete!")
                st.write(f"**Final Order:** {final_order}")
                st.write(f"**Total Time:** {final_time:.2f} seconds")
                st.write(f"**Total Distance:** {final_distance:.2f} meters")
                st.write(f"**Half Laps Completed:** {final_half_lap_count}")
                st.write(f"**Switch at half-laps:** {switch_schedule_description(switch_schedule)}")

                st.subheader("W′ Remaining per Rider:")
                for k, v in W_rem.items():
                    st.write(f"{k}: {v:.1f} J")
