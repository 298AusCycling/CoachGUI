import streamlit as st
import pandas as pd
from iteration import simulate_race

st.title("Team Pursuit Race Simulator")
drafting_percents = [1.0, 0.58, 0.52, 0.53]
lap_list = []
def switch_schedule_description(switch_schedule):
    lap_list = []  
    for i in range(len(switch_schedule)):
        if switch_schedule[i] == 1:
            lap_list.append(i + 1)
    return lap_list

uploaded_file = st.file_uploader("Upload Performance Data Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    power_duration_df = pd.read_excel(uploaded_file, sheet_name="Power Curves")

    available_athletes = df["Name"].str.extract(r'M(\d+)')[0].dropna().astype(int).tolist()
    chosen_athletes = st.segmented_control("Select 4 Athletes", options=available_athletes, selection_mode = 'multi', default=None, key = 1323)
    st.markdown(f"Selected Riders: {chosen_athletes}.")

    if len(chosen_athletes) == 4:
        start_order = st.segmented_control("Initial Rider Order", options=sorted(chosen_athletes), selection_mode = 'multi', default=None, key = 1231)
        st.markdown(f"Initial Starting Order: {start_order}.")

        st.subheader("Switch Schedule (32 half-laps)")
        switch_schedule = []
        peel_schedule = []

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Switch (1 = switch after this half-lap)**")
            for i in range(32):
                switch = st.checkbox(f"{i+1}", key=f"switch_{i}")
                switch_schedule.append(1 if switch else 0)

        with col2:
            st.markdown("**Peel (1 = peel here)**")
            for i in range(32):
                peel = st.checkbox(f"{i+1}", key=f"peel_{i}")
                peel_schedule.append(1 if peel else 0)

        # Determine peel location
        try:
            peel_location = peel_schedule.index(1)  # peel at the first '1'
        except ValueError:
            st.error("⚠️ Please select at least one peel location.")
            st.stop()
        
    


        if st.button("Simulate Race"):
            with st.spinner("Running simulation..."):
                final_order, final_time, final_distance, final_half_lap_count, W_rem = simulate_race(
                    switch_schedule=switch_schedule,
                    chosen_athletes=chosen_athletes,
                    start_order=start_order,
                    drafting_percents=drafting_percents,
                    peel_location=peel_location,
                    power_duration_df=power_duration_df
                )

            st.success("✅ Simulation Complete!")
            st.write(f"**Final Order:** {final_order}")
            st.write(f"**Total Time:** {final_time:.2f} seconds")
            st.write(f"**Total Distance:** {final_distance:.2f} m")
            st.write(f"**Half Laps Completed:** {final_half_lap_count}")
            st.write(f"**Switch at laps** {str(switch_schedule_description(switch_schedule))}")

            st.subheader("W′ Remaining per Rider:")
            for k, v in W_rem.items():
                st.write(f"{k}: {v:.1f} J")
