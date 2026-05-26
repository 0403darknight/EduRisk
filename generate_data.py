import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

age = np.random.randint(17, 55, n)
gender = np.random.choice([0, 1], n)
scholarship = np.random.choice([0, 1], n, p=[0.6, 0.4])
debtor = np.random.choice([0, 1], n, p=[0.8, 0.2])
fees_up_to_date = np.random.choice([0, 1], n, p=[0.25, 0.75])
attendance = np.random.randint(30, 100, n)
sem1_enrolled = np.random.randint(3, 7, n)
sem1_approved = np.clip(sem1_enrolled - np.random.randint(0, 3, n), 0, 7)
sem1_grade = np.random.uniform(8, 18, n)
sem2_enrolled = np.random.randint(3, 7, n)
sem2_approved = np.clip(sem2_enrolled - np.random.randint(0, 3, n), 0, 7)
sem2_grade = sem1_grade + np.random.normal(0, 2, n)
sem2_grade = np.clip(sem2_grade, 0, 20)
admission_grade = np.random.uniform(95, 190, n)
prev_qual_grade = np.random.uniform(90, 190, n)
age_enrollment = age - np.random.randint(0, 5, n)
unemployment_rate = np.random.uniform(6, 16, n)
inflation_rate = np.random.uniform(-0.8, 3.5, n)
gdp = np.random.uniform(-4, 3, n)
marital_status = np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1])
application_mode = np.random.choice(range(1, 18), n)
course = np.random.choice(range(1, 18), n)
day_evening = np.random.choice([0, 1], n, p=[0.7, 0.3])
prev_qualification = np.random.choice(range(1, 12), n)
nationality = np.random.choice(range(1, 22), n)
mothers_qual = np.random.choice(range(1, 7), n)
fathers_qual = np.random.choice(range(1, 7), n)
mothers_occ = np.random.choice(range(0, 10), n)
fathers_occ = np.random.choice(range(0, 10), n)
displaced = np.random.choice([0, 1], n, p=[0.7, 0.3])
special_needs = np.random.choice([0, 1], n, p=[0.95, 0.05])
international = np.random.choice([0, 1], n, p=[0.9, 0.1])

dropout_score = (
    -0.03 * attendance
    - 0.12 * sem1_grade
    - 0.12 * sem2_grade
    - 0.15 * fees_up_to_date
    + 0.10 * debtor
    - 0.08 * scholarship
    + 0.04 * np.abs(sem2_grade - sem1_grade)
    + 0.02 * unemployment_rate
    + 0.003 * age
    + np.random.normal(0, 0.5, n)
)

percentiles = np.percentile(dropout_score, [32, 65])
target = np.where(dropout_score > percentiles[1], 0,
          np.where(dropout_score > percentiles[0], 1, 2))

df = pd.DataFrame({
    'Marital status': marital_status,
    'Application mode': application_mode,
    'Application order': np.random.randint(1, 9, n),
    'Course': course,
    'Daytime/evening attendance': day_evening,
    'Previous qualification': prev_qualification,
    'Previous qualification (grade)': prev_qual_grade,
    'Nationality': nationality,
    "Mother's qualification": mothers_qual,
    "Father's qualification": fathers_qual,
    "Mother's occupation": mothers_occ,
    "Father's occupation": fathers_occ,
    'Admission grade': admission_grade,
    'Displaced': displaced,
    'Educational special needs': special_needs,
    'Debtor': debtor,
    'Tuition fees up to date': fees_up_to_date,
    'Gender': gender,
    'Scholarship holder': scholarship,
    'Age at enrollment': age_enrollment,
    'International': international,
    'Curricular units 1st sem (credited)': np.random.randint(0, 3, n),
    'Curricular units 1st sem (enrolled)': sem1_enrolled,
    'Curricular units 1st sem (evaluations)': sem1_enrolled + np.random.randint(0, 2, n),
    'Curricular units 1st sem (approved)': sem1_approved,
    'Curricular units 1st sem (grade)': sem1_grade,
    'Curricular units 1st sem (without evaluations)': np.random.randint(0, 2, n),
    'Curricular units 2nd sem (credited)': np.random.randint(0, 3, n),
    'Curricular units 2nd sem (enrolled)': sem2_enrolled,
    'Curricular units 2nd sem (evaluations)': sem2_enrolled + np.random.randint(0, 2, n),
    'Curricular units 2nd sem (approved)': sem2_approved,
    'Curricular units 2nd sem (grade)': sem2_grade,
    'Curricular units 2nd sem (without evaluations)': np.random.randint(0, 2, n),
    'Unemployment rate': unemployment_rate,
    'Inflation rate': inflation_rate,
    'GDP': gdp,
    'Attendance (%)': attendance,
    'Target': target
})

df.to_csv('data/dataset.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df['Target'].value_counts().rename({0:'Dropout',1:'Enrolled',2:'Graduate'}))
