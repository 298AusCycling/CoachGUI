import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.title("AusCycling Performance Analysis")
    
    # Rider Selection
    st.sidebar.header("Select Riders")
    riders = ["Rider 1", "Rider 2", "Rider 3", "Rider 4"]  # Placeholder
    selected_riders = st.sidebar.multiselect("Choose riders to compare:", riders)
    
    # Physiological Inputs for Each Selected Rider
    st.sidebar.header("Physiological Inputs")
    rider_data = {}
    for rider in selected_riders:
        st.sidebar.subheader(f"{rider} Data")
        rider_data[rider] = {
            "Weight (kg)": st.sidebar.number_input(f"{rider} - Weight (kg)", min_value=40, max_value=120, value=70, key=f"weight_{rider}"),
            "Power Output (W)": st.sidebar.number_input(f"{rider} - Power Output (W)", min_value=100, max_value=2000, value=250, key=f"power_{rider}"),
            "Drag Coefficient (CdA)": st.sidebar.number_input(f"{rider} - Drag Coefficient (CdA)", min_value=0.1, max_value=0.4, value=0.25, key=f"cda_{rider}"),
            "Rolling Resistance": st.sidebar.number_input(f"{rider} - Rolling Resistance", min_value=0.002, max_value=0.01, value=0.004, key=f"rr_{rider}"),
            "Air Density (kg/m³)": st.sidebar.number_input(f"{rider} - Air Density (kg/m³)", min_value=1.0, max_value=1.3, value=1.225, key=f"air_{rider}"),
            "Track Slope (%)": st.sidebar.slider(f"{rider} - Track Slope (%)", min_value=-5.0, max_value=5.0, value=0.0, key=f"slope_{rider}")
        }
    
    
    # File Upload for Athlete Data
    uploaded_file = st.file_uploader("Upload Athlete Data (CSV, JSON, etc.)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        # Ensure required columns exist
        if all(col in df.columns for col in ["Name", "CP", "W", "Pmax"]):
            st.write("### Power Curve Visualization")
            
            t = np.linspace(1, 100, 100000)  # Time range
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for _, row in df.iterrows():
                P = row['W'] / t + row['CP']
                ax.plot(t, P, label=f"{row['Name']} - W={row['W']}, CP={row['CP']}")
            
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("Power (P)")
            ax.set_title("Power Curve for Riders")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.error("Uploaded file does not contain required columns: 'Name', 'CP', 'W', 'Pmax'")
    
    # Submit Button
    if st.button("Submit"):
        if not selected_riders:
            st.warning("Please select at least one rider.")
        else:
            st.success("Processing data for selected riders...")
            # Display entered data
            st.write("### Entered Data for Selected Riders")
            data = []
            for rider, values in rider_data.items():
                st.write(f"#### {rider}")
                for key, value in values.items():
                    st.write(f"{key}: {value}")
                data.append([rider] + list(values.values()))
            
            # Convert data to DataFrame for visualization
            columns = ["Rider"] + list(rider_data[selected_riders[0]].keys())
            df = pd.DataFrame(data, columns=columns)
            df.set_index("Rider", inplace=True)
            
            # Generate graphs
            st.write("### Data Visualization")
            for column in df.columns:
                fig, ax = plt.subplots()
                df[column].plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                plt.ylabel(column)
                plt.title(f"Comparison of {column} among Riders")
                st.pyplot(fig)

if __name__ == "__main__":
    main()