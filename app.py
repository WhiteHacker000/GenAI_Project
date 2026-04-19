import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="EV Charging Station Energy Prediction",
    page_icon="⚡",
    layout="wide"
)

@st.cache_resource
def load_model():
    if os.path.exists('model_artifacts.pkl'):
        return joblib.load('model_artifacts.pkl')
    return None

def main():
    # Hardcoded API Key so users don't have to enter it in the UI!
    groq_api_key = os.getenv("GROQ_API_KEY", "PASTE_YOUR_API_KEY_HERE")
    
    st.title("⚡ EV Charging Station Energy Prediction")
    st.markdown("Predict the total energy (kWh) consumed at a specific charging station based on the time of day and day of the week.")
    
    artifacts = load_model()
    
    if artifacts is None:
        st.error("Model artifacts not found! Please run `python train_model.py` first to generate the required `model_artifacts.pkl` file.")
        return
        
    rf_model = artifacts['model']
    station_avg_map = artifacts['station_avg_map']
    feature_columns = artifacts['feature_columns']
    
    # Extract unique station names, they are keys in the avg map
    stations = sorted(list(station_avg_map.keys()))
    
    # UI Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        selected_station = st.selectbox(
            "Select Charging Station",
            options=stations,
            help="Choose the station to predict energy for."
        )
        
        selected_hour = st.slider(
            "Select Hour of the Day",
            min_value=0,
            max_value=23,
            value=12,
            step=1,
            help="0 is Midnight, 12 is Noon."
        )
        
        # Mapping for display vs internal int representation
        weekdays = {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 
            3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
        }
        
        selected_weekday_name = st.selectbox(
            "Select Day of the Week",
            options=list(weekdays.values())
        )
        
        # Get the integer key for the selected string value
        selected_weekday = list(weekdays.keys())[list(weekdays.values()).index(selected_weekday_name)]
        
        if st.button("Predict 🔮", type="primary", use_container_width=True):
            with st.spinner('Calculating prediction...'):
                # Build the input features dictionary
                input_data = {
                    'Hour': selected_hour,
                    'Weekday': selected_weekday
                }
                
                # Computed features
                input_data['Peak_Hour'] = 1 if (6 <= selected_hour <= 10) or (17 <= selected_hour <= 21) else 0
                input_data['Is_Weekend'] = 1 if selected_weekday >= 5 else 0
                input_data['Station_Avg_Load'] = station_avg_map.get(selected_station, 0)
                input_data['Hour_Weekend_Interaction'] = input_data['Hour'] * input_data['Is_Weekend']
                
                df_input = pd.DataFrame([input_data])
                model_input = pd.DataFrame(columns=feature_columns)
                model_input.loc[0] = 0
                
                for col in ['Hour', 'Weekday', 'Peak_Hour', 'Is_Weekend', 'Station_Avg_Load', 'Hour_Weekend_Interaction']:
                    if col in feature_columns:
                        model_input.at[0, col] = df_input.at[0, col]
                        
                station_col = f"Station Name_{selected_station}"
                if station_col in feature_columns:
                    model_input.at[0, station_col] = 1
                
                model_input = model_input.infer_objects()
                prediction = rf_model.predict(model_input)[0]
                
                # Save to session state for the agent
                st.session_state['last_prediction'] = prediction
                st.session_state['last_station'] = selected_station
                st.session_state['last_time_context'] = f"{selected_hour}:00 on {selected_weekday_name}"
                st.session_state['last_avg'] = station_avg_map.get(selected_station, 0)
                
            # Render prediction results nicely
            with col2:
                st.subheader("Prediction Result")
                st.metric(
                    label=f"Predicted Energy roughly around {selected_hour}:00",
                    value=f"{prediction:.2f} kWh"
                )
                
                st.info(f"**Baseline Context:** The historical average load for this station across all times is **{station_avg_map.get(selected_station, 0):.2f} kWh**.")
                
                if prediction > station_avg_map.get(selected_station, 0):
                    st.warning("📈 Note: Predicted usage is above the station average. Expect higher demand.")
                else:
                    st.success("📉 Note: Predicted usage is below or near the station average. Demand should be normal.")
                    
        # --- MILESTONE 2: AGENTIC WORKFLOW ---
        st.divider()
        st.header("🤖 Agentic Infrastructure Planning (Milestone 2)")
        
        if st.button("Run LangGraph Agent", type="secondary", use_container_width=True):
            if 'last_prediction' not in st.session_state:
                st.error("Please run the Predict 🔮 step first!")
            else:
                from agent import run_agentic_workflow
                
                with st.spinner("Agent compiling plan and reasoning over guidelines..."):
                    final_plan = run_agentic_workflow(
                        station=st.session_state['last_station'],
                        time_context=st.session_state['last_time_context'],
                        predicted=st.session_state['last_prediction'],
                        avg=st.session_state['last_avg'],
                        groq_key=groq_api_key
                    )
                    
                    if "error" in final_plan:
                        st.error(f"Agent Error: {final_plan['error']}")
                    else:
                        st.success("Agent successfully generated infrastructure plan!")
                        st.markdown("### Agentic Output")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.info(f"**Analysis:** {final_plan.get('Analysis', 'N/A')}")
                            st.info(f"**Locate:** {final_plan.get('Locate', 'N/A')}")
                        with c2:
                            st.success(f"**Plan:** {final_plan.get('Plan', 'N/A')}")
                            st.warning(f"**Optimize:** {final_plan.get('Optimize', 'N/A')}")
                        
                        st.markdown(f"**Refs:** {final_plan.get('Refs', 'N/A')}")

if __name__ == "__main__":
    main()
