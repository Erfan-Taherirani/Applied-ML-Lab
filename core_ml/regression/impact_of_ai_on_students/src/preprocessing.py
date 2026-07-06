"""
This module includes custome functions used for preprocessing in this
project.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def process_data(raw_df: pd.DataFrame):
    raw_df['GPA_difference'] = raw_df[
        'Post_Semester_GPA'] - raw_df['Pre_Semester_GPA']
    
    features_in_lower_case = [feature.lower() for feature in raw_df.columns]
    df = pd.DataFrame(
        data=raw_df.values,
        columns=features_in_lower_case
    )

    # fixing datatypes (categorical features)
    df['major_category'] = df[
        'major_category'].str.lower().astype("category")
    df['year_of_study'] = df[
        'year_of_study'].str.lower().astype("category")
    df['primary_use_case'] = df[
        'primary_use_case'].str.lower().astype("category")
    df['prompt_engineering_skill'] = df[
        'prompt_engineering_skill'].str.lower().astype("category")
    df['institutional_policy'] = df[
        'institutional_policy'].str.lower().astype("category")
    df['burnout_risk_level'] = df[
        'burnout_risk_level'].str.lower().astype("category")
    
    # fixing datatypes (numerical features)
    df['weekly_genai_hours'] = df['weekly_genai_hours'].astype(float)
    df['tool_diversity'] = df['tool_diversity'].astype(int)
    df['paid_subscription'] = df['paid_subscription'].astype(bool)
    df['traditional_study_hours'] = df['traditional_study_hours'].astype(float)
    df['perceived_ai_dependency'] = df['perceived_ai_dependency'].astype(int)
    df['anxiety_level_during_exams'] = df['anxiety_level_during_exams'].astype(int)
    df['skill_retention_score'] = df['skill_retention_score'].astype(float)
    df['gpa_difference'] = df['gpa_difference'].astype(float)
    
    return df


def separate_features(df):
    categorical_features = [
        "major_category",
        "year_of_study",
        "primary_use_case",
        "prompt_engineering_skill",
        "institutional_policy",
        "burnout_risk_level"
    ]
    numerical_features = [
        feature for feature in df.columns if feature not in categorical_features]

    return categorical_features, numerical_features
