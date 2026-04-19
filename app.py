import streamlit as st
import requests
import os

st.set_page_config(
    page_title="EV Charging Station Energy Prediction",
    page_icon="⚡",
    layout="wide"
)

# Backend URL - internally within the Docker container, use 127.0.0.1
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

@st.cache_data
def get_stations():
    try:
        response = requests.get(f"{BACKEND_URL}/stations")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Could not connect to Backend: {e}")
        return []

def main():
    st.title("⚡ EV Charging Station Energy Prediction")
    st.markdown("Predict the total energy (kWh) consumed at a specific charging station based on the time of day and day of the week.")
    
    stations = get_stations()
    
    if not stations:
        st.warning("⚠️ Backend server is not responding. Please make sure the FastAPI server is running.")
        return
        
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
        
        weekdays = {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 
            3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
        }
        
        selected_weekday_name = st.selectbox(
            "Select Day of the Week",
            options=list(weekdays.values())
        )
        
        selected_weekday = list(weekdays.keys())[list(weekdays.values()).index(selected_weekday_name)]
        
        if st.button("Predict 🔮", type="primary", use_container_width=True):
            with st.spinner('Requesting prediction from backend...'):
                payload = {
                    "station_name": selected_station,
                    "hour": selected_hour,
                    "weekday": selected_weekday,
                    "weekday_name": selected_weekday_name
                }
                
                try:
                    response = requests.post(f"{BACKEND_URL}/predict", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        prediction = data['prediction']
                        station_avg = data['station_avg']
                        
                        # Save to session state for the agent
                        st.session_state['last_prediction'] = prediction
                        st.session_state['last_station'] = selected_station
                        st.session_state['last_time_context'] = f"{selected_hour}:00 on {selected_weekday_name}"
                        st.session_state['last_avg'] = station_avg
                        
                        # Render results
                        with col2:
                            st.subheader("Prediction Result")
                            st.metric(
                                label=f"Predicted Energy roughly around {selected_hour}:00",
                                value=f"{prediction:.2f} kWh"
                            )
                            st.info(f"**Baseline Context:** Historical average load for this station is **{station_avg:.2f} kWh**.")
                            
                            if prediction > station_avg:
                                st.warning("📈 Note: Predicted usage is above average. Expect higher demand.")
                            else:
                                st.success("📉 Note: Predicted usage is below or near average. Demand should be normal.")
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    # --- MILESTONE 2: AGENTIC WORKFLOW ---
    st.divider()
    st.header("🤖 Agentic Infrastructure Planning (Milestone 2)")
    
    if st.button("Run LangGraph Agent", type="secondary", use_container_width=True):
        if 'last_prediction' not in st.session_state:
            st.error("Please run the Predict 🔮 step first!")
        else:
            with st.spinner("Agent compiling plan via FastAPI backend..."):
                payload = {
                    "station_name": st.session_state['last_station'],
                    "time_context": st.session_state['last_time_context'],
                    "predicted_energy": st.session_state['last_prediction'],
                    "avg_load": st.session_state['last_avg']
                }
                
                try:
                    response = requests.post(f"{BACKEND_URL}/plan", json=payload)
                    if response.status_code == 200:
                        final_plan = response.json()
                        
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
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

if __name__ == "__main__":
    main()
