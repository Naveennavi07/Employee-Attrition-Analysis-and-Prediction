import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# 1. Load Data and Model Artifacts
# -----------------------------
try:
    # Load the pre-trained pipeline and feature columns
    pipeline = joblib.load('D:\Employee_attrition\env\Scripts\lr_model_pipeline.pkl')
    training_cols = joblib.load('D:\Employee_attrition\env\Scripts\Training_columns.pkl')
    df = pd.read_csv('D:\Employee_attrition\env\Scripts\Employeedata.csv')
except FileNotFoundError:
    st.error("Error: Required model and data files not found.")
    st.error("Please ensure 'lr_model_pipeline.pkl', 'training_columns.pkl', and 'Employee_data.csv' are in the same directory.")
    st.stop()


# -----------------------------
# 2. Sidebar Navigation
# -----------------------------
st.sidebar.title("Employee Attrition Analysis")
page = st.sidebar.radio("Navigation", ["Home", "Predict Attrition"])


# -----------------------------
# 3. Home Dashboard Page
# -----------------------------
if page == "Home":
    st.markdown("<h1 style='text-align:center;'>Employee Insights Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("This dashboard provides a quick overview of key employee data.")

    # High Risk Employees
    st.markdown("### 🚨 High Risk Employees (Top 5)")
    high_risk = df[df["Attrition"] == "Yes"].head(5)
    st.dataframe(high_risk[["EmployeeNumber", "Age", "MonthlyIncome", "JobRole", "Attrition"]])

    # High Job Satisfaction
    st.markdown("### 😀 High Job Satisfaction (Top 5)")
    high_satisfaction = df.sort_values("JobSatisfaction", ascending=False).head(5)
    st.dataframe(high_satisfaction[["EmployeeNumber", "JobRole", "JobSatisfaction", "MonthlyIncome"]])


# -----------------------------
# 4. Attrition Prediction Page
# -----------------------------
# -----------------------------
# -----------------------------
# 4. Attrition Prediction Page (Most Robust Version)
# -----------------------------
elif page == "Predict Attrition":
    st.markdown("<h1 style='text-align:center;'>Predict Employee Attrition</h1>", unsafe_allow_html=True)

    with st.form(key='prediction_form'):
        st.markdown("### Please enter the employee's details:")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=65, step=1, value=30)
            monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, step=500, value=6000)
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, step=1, value=5)
            stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        
        with col2:
            job_role = st.selectbox("Job Role", sorted([
                "Sales Executive", "Research Scientist", "Laboratory Technician", 
                "Manufacturing Director", "Healthcare Representative", "Manager", 
                "Sales Representative", "Human Resources", "Research Director"
            ]))
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            overtime = st.selectbox("OverTime", ["Yes", "No"])
        
        submit_button = st.form_submit_button(label="Run Prediction")

    if submit_button:
        try:
            # Create a base DataFrame with the correct numerical columns
            # The order of these columns is critical.
            input_df = pd.DataFrame([{
                'Age': age,
                'DailyRate': 0, # Placeholder, not in user input
                'DistanceFromHome': 0, # Placeholder, not in user input
                'Education': 0, # Placeholder, not in user input
                'EnvironmentSatisfaction': 0, # Placeholder, not in user input
                'HourlyRate': 0, # Placeholder, not in user input
                'JobInvolvement': 0, # Placeholder, not in user input
                'JobLevel': 0, # Placeholder, not in user input
                'JobSatisfaction': 0, # Placeholder, not in user input
                'MonthlyIncome': monthly_income,
                'MonthlyRate': 0, # Placeholder, not in user input
                'NumCompaniesWorked': 0, # Placeholder, not in user input
                'PercentSalaryHike': 0, # Placeholder, not in user input
                'PerformanceRating': 0, # Placeholder, not in user input
                'RelationshipSatisfaction': 0, # Placeholder, not in user input
                'StockOptionLevel': stock_option,
                'TotalWorkingYears': 0, # Placeholder, not in user input
                'TrainingTimesLastYear': 0, # Placeholder, not in user input
                'WorkLifeBalance': 0, # Placeholder, not in user input
                'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': 0, # Placeholder, not in user input
                'YearsSinceLastPromotion': 0, # Placeholder, not in user input
                'YearsWithCurrManager': 0 # Placeholder, not in user input
            }])

            # Add one-hot encoded columns, pre-filled with 0s
            # Assuming alphabetical sorting for drop_first=True
            input_df['OverTime_Yes'] = 1 if overtime == 'Yes' else 0
            
            input_df['BusinessTravel_Travel_Frequently'] = 0
            input_df['BusinessTravel_Travel_Rarely'] = 0
            
            input_df['Department_Research & Development'] = 0
            input_df['Department_Sales'] = 0

            input_df['EducationField_Life Sciences'] = 0
            input_df['EducationField_Marketing'] = 0
            input_df['EducationField_Medical'] = 0
            input_df['EducationField_Other'] = 0
            input_df['EducationField_Technical Degree'] = 0

            input_df['Gender_Male'] = 0

            if marital_status == 'Single':
                input_df['MaritalStatus_Single'] = 1
            elif marital_status == 'Married':
                input_df['MaritalStatus_Married'] = 1
            elif marital_status == 'Divorced':
                input_df['MaritalStatus_Divorced'] = 1

            # Manually set the one-hot encoded column for Job Role
            job_role_col_name = f'JobRole_{job_role}'
            input_df[job_role_col_name] = 1

            # Ensure the DataFrame has the exact same columns and order as the training data
            
            input_df = input_df.reindex(columns=training_cols, fill_value=0)

            # Make predictions
            prediction = pipeline.predict(input_df)[0]
            probability = pipeline.predict_proba(input_df)[0]

            # Display the result
            st.markdown("### Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ **Prediction: Employee is likely to leave (Attrition)**")
                st.write(f"Probability of leaving: **{probability[1]:.2f}**")
            else:
                st.success(f"✅ **Prediction: Employee is likely to stay**")
                st.write(f"Probability of staying: **{probability[0]:.2f}**")

        except Exception as e:
            st.warning(f"Error during prediction: {e}")