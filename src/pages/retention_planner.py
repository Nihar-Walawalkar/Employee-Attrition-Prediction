"""
Retention Planner page for creating and managing employee retention strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from src.utils import data_loader, config
from src.utils.data_loader import safe_display_dataframe

def show():
    """Display the retention planner page."""
    
    st.markdown("<h1 style='text-align: center; color: #4ecdc4;'>Retention Planner</h1>", unsafe_allow_html=True)
    st.write("Create and track personalized retention strategies for high-risk employees.")

    # Check if model exists to provide risk assessment
    model_exists = all([
        os.path.exists(config.BEST_MODEL_PATH),
        os.path.exists(config.SCALER_PATH),
        os.path.exists(config.FEATURES_PATH)
    ])
    
    # Load the dataset
    df = data_loader.load_dataset()
    
    # Initialize session state for retention plans if not exists
    if 'retention_plans' not in st.session_state:
        # Load from file if exists
        plan_file = os.path.join(config.ARTIFACTS_DIR, "retention_plans.json")
        if os.path.exists(plan_file):
            try:
                with open(plan_file, 'r') as f:
                    st.session_state.retention_plans = json.load(f)
            except Exception as e:
                st.warning(f"Could not load existing plans: {str(e)}")
                st.session_state.retention_plans = {}
        else:
            st.session_state.retention_plans = {}
    
    # Create tabs for different sections
    tabs = st.tabs(["Dashboard", "Create Plan", "Track Progress"])
    
    # Dashboard tab
    with tabs[0]:
        st.subheader("Retention Strategy Dashboard")
        
        if not st.session_state.retention_plans:
            st.info("No retention plans have been created yet. Go to 'Create Plan' to create your first retention strategy.")
        else:
            try:
                # Summary stats
                total_plans = len(st.session_state.retention_plans)
                active_plans = sum(1 for plan in st.session_state.retention_plans.values() 
                                if plan.get('status', '') == 'Active')
                completed_plans = sum(1 for plan in st.session_state.retention_plans.values() 
                                    if plan.get('status', '') == 'Completed')
                success_plans = sum(1 for plan in st.session_state.retention_plans.values() 
                                if plan.get('outcome', '') == 'Successful')
                
                # Display stats in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Plans", total_plans)
                
                with col2:
                    st.metric("Active Plans", active_plans)
                
                with col3:
                    st.metric("Completed", completed_plans)
                
                with col4:
                    success_rate = f"{(success_plans/completed_plans*100):.1f}%" if completed_plans > 0 else "N/A"
                    st.metric("Success Rate", success_rate)
                
                # Display all plans in a table
                st.write("### All Retention Plans")
                
                # Convert plans to DataFrame for display
                plans_data = []
                for emp_id, plan in st.session_state.retention_plans.items():
                    plans_data.append({
                        'Employee ID': emp_id,
                        'Name': plan.get('name', 'Unknown'),
                        'Risk Level': plan.get('risk_level', 'Unknown'),
                        'Start Date': plan.get('start_date', ''),
                        'Status': plan.get('status', 'Active'),
                        'Strategy Type': plan.get('strategy_type', ''),
                        'Outcome': plan.get('outcome', 'Pending')
                    })
                
                if plans_data:
                    safe_display_dataframe(plans_data)
                    
                    # Add visualizations (only if we have data)
                    if plans_data:
                        try:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Plans by risk level
                                plans_df = pd.DataFrame(plans_data)
                                risk_counts = plans_df['Risk Level'].value_counts().reset_index()
                                risk_counts.columns = ['Risk Level', 'Count']
                                
                                fig = px.pie(
                                    risk_counts, 
                                    values='Count', 
                                    names='Risk Level',
                                    title='Plans by Risk Level',
                                    color='Risk Level',
                                    color_discrete_map={
                                        'High': '#ff6f61',
                                        'Medium': '#ffb347',
                                        'Low': '#4ecdc4',
                                        'Unknown': '#cccccc'
                                    },
                                    hole=0.4
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Plans by strategy type
                                if 'Strategy Type' in plans_df.columns and not plans_df['Strategy Type'].isnull().all():
                                    strategy_counts = plans_df['Strategy Type'].value_counts().reset_index()
                                    strategy_counts.columns = ['Strategy Type', 'Count']
                                    
                                    fig = px.bar(
                                        strategy_counts,
                                        x='Strategy Type',
                                        y='Count',
                                        title='Plans by Strategy Type',
                                        color='Count',
                                        color_continuous_scale='Teal'
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info("No strategy type data available for visualization.")
                        except Exception as e:
                            st.error(f"Error creating visualizations: {str(e)}")
                else:
                    st.info("No plan data available to display.")
            except Exception as e:
                st.error(f"Error displaying dashboard: {str(e)}")
    
    # Create Plan tab
    with tabs[1]:
        st.subheader("Create New Retention Plan")
        
        if df is None or df.empty:
            st.error("No employee data available. Cannot create retention plans.")
        else:
            try:
                # Display employee selector
                employee_list = []
                if 'EmployeeNumber' in df.columns:
                    employee_list = df['EmployeeNumber'].astype(str).tolist()
                elif 'EmployeeID' in df.columns:
                    employee_list = df['EmployeeID'].astype(str).tolist()
                else:
                    # If no employee ID column, create a list of indices
                    employee_list = [str(i) for i in range(1, len(df) + 1)]
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    selected_employee = st.selectbox("Select Employee", employee_list)
                    
                    # Extract employee data
                    if selected_employee and not df.empty:
                        # Find the employee data based on whatever ID column we used
                        if 'EmployeeNumber' in df.columns:
                            employee_data = df[df['EmployeeNumber'].astype(str) == selected_employee]
                        elif 'EmployeeID' in df.columns:
                            employee_data = df[df['EmployeeID'].astype(str) == selected_employee]
                        else:
                            # Use index as fallback
                            try:
                                idx = int(selected_employee) - 1
                                if 0 <= idx < len(df):
                                    employee_data = df.iloc[[idx]]
                                else:
                                    employee_data = pd.DataFrame()
                            except:
                                employee_data = pd.DataFrame()
                        
                        if not employee_data.empty:
                            employee_data = employee_data.iloc[0]
                            # Risk assessment
                            if model_exists:
                                risk_score = np.random.random()  # Placeholder - would use actual model
                                risk_level = "High" if risk_score > 0.66 else "Medium" if risk_score > 0.33 else "Low"
                            else:
                                risk_score = np.random.random()  # Random placeholder
                                risk_level = "High" if risk_score > 0.66 else "Medium" if risk_score > 0.33 else "Low"
                        else:
                            st.error("Employee data not found")
                            risk_level = "Unknown"
                            risk_score = 0
                
                with col2:
                    if selected_employee and not df.empty and not employee_data.empty:
                        # Display employee info in a card
                        # Try different field combinations for employee name
                        if 'FirstName' in employee_data and 'LastName' in employee_data:
                            employee_name = f"{employee_data.get('FirstName', '')} {employee_data.get('LastName', '')}"
                        elif 'First Name' in employee_data and 'Last Name' in employee_data:
                            employee_name = f"{employee_data.get('First Name', '')} {employee_data.get('Last Name', '')}"
                        elif 'Name' in employee_data:
                            employee_name = employee_data.get('Name', '')
                        else:
                            employee_name = f"Employee {selected_employee}"
                        
                        # Get department info, checking different possible column names
                        if 'Department' in employee_data:
                            department = employee_data.get('Department', 'Unknown')
                        else:
                            department = 'Unknown'
                        
                        # Get job role info, checking different possible column names
                        if 'JobRole' in employee_data:
                            job_role = employee_data.get('JobRole', 'Unknown')
                        elif 'Job Role' in employee_data:
                            job_role = employee_data.get('Job Role', 'Unknown')
                        elif 'Position' in employee_data:
                            job_role = employee_data.get('Position', 'Unknown')
                        else:
                            job_role = 'Unknown'
                        
                        st.markdown(
                            f"""
                            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
                                <h3 style="margin-top: 0;">{employee_name}</h3>
                                <p><strong>Department:</strong> {department}</p>
                                <p><strong>Job Role:</strong> {job_role}</p>
                                <p><strong>Risk Level:</strong> <span style="color: {'#ff6f61' if risk_level == 'High' else '#ffb347' if risk_level == 'Medium' else '#4ecdc4'}; font-weight: bold;">{risk_level}</span></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                # Plan details
                if selected_employee:
                    st.write("### Plan Details")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        strategy_type = st.selectbox(
                            "Strategy Type",
                            ["Compensation Adjustment", "Career Development", "Work-Life Balance", 
                             "Role Redesign", "Recognition Program", "Learning Opportunity"]
                        )
                        
                        priority = st.selectbox("Priority", ["High", "Medium", "Low"])
                        
                        start_date = st.date_input("Start Date", datetime.now())
                        end_date = st.date_input("Target End Date", datetime.now() + timedelta(days=90))
                    
                    with col2:
                        plan_description = st.text_area(
                            "Plan Description", 
                            placeholder="Describe the retention strategy and specific actions to take..."
                        )
                        
                        success_criteria = st.text_area(
                            "Success Criteria",
                            placeholder="What measurable outcomes will indicate success?"
                        )
                        
                        stakeholders = st.text_input(
                            "Stakeholders",
                            placeholder="List key stakeholders involved (e.g., HR Manager, Department Head)"
                        )
                    
                    # Recommended strategies based on risk factors
                    st.write("### Recommended Strategies")
                    
                    # These would be generated from actual model insights, but using placeholders here
                    if risk_level == "High":
                        recommendations = [
                            "**Compensation Review**: Consider a salary adjustment based on market rates and internal equity",
                            "**Career Path Discussion**: Schedule 1-on-1 meeting to discuss growth opportunities",
                            "**Workload Assessment**: Evaluate current workload and adjust if necessary"
                        ]
                    elif risk_level == "Medium":
                        recommendations = [
                            "**Skill Development**: Identify training opportunities aligned with career goals",
                            "**Recognition Program**: Enroll in the quarterly recognition program",
                            "**Mentorship**: Assign a senior mentor from the same department"
                        ]
                    else:
                        recommendations = [
                            "**Regular Check-ins**: Schedule monthly informal feedback sessions",
                            "**Team Engagement**: Include in cross-departmental projects",
                            "**Work Flexibility**: Consider flexible work arrangements if appropriate"
                        ]
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    
                    # Save button
                    if st.button("Create Retention Plan"):
                        if not plan_description:
                            st.error("Please provide a plan description before saving")
                        else:
                            # Create plan object
                            new_plan = {
                                "name": employee_name,
                                "risk_level": risk_level,
                                "risk_score": float(risk_score),
                                "department": department,
                                "job_role": job_role,
                                "strategy_type": strategy_type,
                                "priority": priority,
                                "start_date": start_date.strftime("%Y-%m-%d"),
                                "end_date": end_date.strftime("%Y-%m-%d"),
                                "description": plan_description,
                                "success_criteria": success_criteria,
                                "stakeholders": stakeholders,
                                "status": "Active",
                                "progress": 0,
                                "outcome": "Pending",
                                "notes": [],
                                "created_date": datetime.now().strftime("%Y-%m-%d"),
                                "last_updated": datetime.now().strftime("%Y-%m-%d")
                            }
                            
                            # Save to session state
                            st.session_state.retention_plans[selected_employee] = new_plan
                            
                            # Save to file
                            plan_file = os.path.join(config.ARTIFACTS_DIR, "retention_plans.json")
                            os.makedirs(os.path.dirname(plan_file), exist_ok=True)
                            with open(plan_file, 'w') as f:
                                json.dump(st.session_state.retention_plans, f, indent=2)
                            
                            st.success("Retention plan created successfully!")
                            st.balloons()
            except Exception as e:
                st.error(f"Error in create plan section: {str(e)}")
    
    # Track Progress tab
    with tabs[2]:
        st.subheader("Track Plan Progress")
        
        if not st.session_state.retention_plans:
            st.info("No retention plans have been created yet. Go to 'Create Plan' to create your first retention strategy.")
        else:
            try:
                # Employee selector for tracking
                plan_options = {emp_id: f"{plan.get('name', 'Unknown')} ({plan.get('risk_level', 'Unknown')} Risk)" 
                              for emp_id, plan in st.session_state.retention_plans.items()}
                
                selected_plan_id = st.selectbox(
                    "Select Plan to Track", 
                    list(plan_options.keys()),
                    format_func=lambda x: plan_options[x]
                )
                
                if selected_plan_id:
                    plan = st.session_state.retention_plans[selected_plan_id]
                    
                    # Display plan details
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("### Plan Details")
                        st.write(f"**Employee:** {plan.get('name', 'Unknown')}")
                        st.write(f"**Strategy:** {plan.get('strategy_type', 'N/A')}")
                        st.write(f"**Department:** {plan.get('department', 'N/A')}")
                        st.write(f"**Job Role:** {plan.get('job_role', 'N/A')}")
                        st.write(f"**Start Date:** {plan.get('start_date', 'N/A')}")
                        st.write(f"**Target End Date:** {plan.get('end_date', 'N/A')}")
                        st.write(f"**Description:** {plan.get('description', 'N/A')}")
                    
                    with col2:
                        # Progress tracking
                        current_progress = plan.get('progress', 0)
                        current_status = plan.get('status', 'Active')
                        current_outcome = plan.get('outcome', 'Pending')
                        
                        # Only allow updates if plan is active
                        if current_status == 'Active':
                            new_progress = st.slider("Progress", 0, 100, current_progress, 5)
                            new_status = st.selectbox("Status", ["Active", "Completed"], index=0 if current_status == "Active" else 1)
                            
                            # Only show outcome selector if status is Completed
                            if new_status == "Completed":
                                new_outcome = st.selectbox("Outcome", ["Successful", "Unsuccessful"])
                            else:
                                new_outcome = "Pending"
                            
                            # Notes
                            new_note = st.text_area("Progress Note", placeholder="Add a note about current progress...")
                            
                            if st.button("Update Progress"):
                                try:
                                    # Update the plan
                                    st.session_state.retention_plans[selected_plan_id]['progress'] = new_progress
                                    st.session_state.retention_plans[selected_plan_id]['status'] = new_status
                                    st.session_state.retention_plans[selected_plan_id]['outcome'] = new_outcome
                                    st.session_state.retention_plans[selected_plan_id]['last_updated'] = datetime.now().strftime("%Y-%m-%d")
                                    
                                    # Add note if provided
                                    if new_note:
                                        if 'notes' not in st.session_state.retention_plans[selected_plan_id]:
                                            st.session_state.retention_plans[selected_plan_id]['notes'] = []
                                        
                                        st.session_state.retention_plans[selected_plan_id]['notes'].append({
                                            'date': datetime.now().strftime("%Y-%m-%d"),
                                            'note': new_note
                                        })
                                    
                                    # Save to file
                                    plan_file = os.path.join(config.ARTIFACTS_DIR, "retention_plans.json")
                                    os.makedirs(os.path.dirname(plan_file), exist_ok=True)
                                    with open(plan_file, 'w') as f:
                                        json.dump(st.session_state.retention_plans, f, indent=2)
                                    
                                    st.success("Progress updated successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating progress: {str(e)}")
                        else:
                            st.info(f"This plan is marked as {current_status}. No further updates are possible.")
                    
                    # Progress visualization
                    try:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=current_progress,
                            title={'text': "Plan Progress"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#4ecdc4"},
                                'steps': [
                                    {'range': [0, 33], 'color': "#f8f9fa"},
                                    {'range': [33, 66], 'color': "#e6f7f5"},
                                    {'range': [66, 100], 'color': "#d0f0ed"}
                                ],
                                'threshold': {
                                    'line': {'color': "green", 'width': 2},
                                    'thickness': 0.75,
                                    'value': 100
                                }
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying progress gauge: {str(e)}")
                    
                    # Notes history
                    st.write("### Progress Notes")
                    notes = plan.get('notes', [])
                    
                    if notes:
                        try:
                            # Format notes for display
                            notes_data = [
                                {"Date": note.get("date", ""), "Note": note.get("note", "")} 
                                for note in notes
                            ]
                            safe_display_dataframe(notes_data, hide_index=True)
                        except Exception as e:
                            st.error(f"Error displaying notes: {str(e)}")
                            st.write("Raw notes data:")
                            for note in notes:
                                st.write(f"- {note.get('date', '')}: {note.get('note', '')}")
                    else:
                        st.info("No progress notes have been added yet.")
            except Exception as e:
                st.error(f"Error in track progress section: {str(e)}") 