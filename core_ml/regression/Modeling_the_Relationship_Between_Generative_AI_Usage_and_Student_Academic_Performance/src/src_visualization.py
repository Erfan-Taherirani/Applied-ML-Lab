"""
This module includes visualization functions that needed in the project.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_univariate_distributions(df):
    _, axs = plt.subplots(2, 3, figsize=(12, 8))
    plt.suptitle("Distributions of the Numerical Features")

    sns.histplot(data=df, x="pre_semester_gpa", kde=True, stat="percent", bins=30, ax=axs.ravel()[0])
    axs.ravel()[0].set_title("Pre Semester GPA Distribution")
    
    sns.histplot(data=df, x="weekly_genai_hours", kde=True, stat="percent", bins=30, ax=axs.ravel()[1])
    axs.ravel()[1].set_title("Weekly GenAI Hours Distribution")
    
    sns.histplot(data=df, x="traditional_study_hours", kde=True, stat="percent", bins=30, ax=axs.ravel()[2])
    axs.ravel()[2].set_title("Traditional Studey Hours Distribution")

    sns.histplot(data=df, x="post_semester_gpa", kde=True, stat="percent", bins=30, ax=axs.ravel()[3])
    axs.ravel()[3].set_title("Post Semester GPA Distribution")
    
    sns.histplot(data=df, x="skill_retention_score", kde=True, stat="percent", bins=30, ax=axs.ravel()[4])
    axs.ravel()[4].set_title("Skill Retention Score Distribution")
    
    sns.histplot(data=df, x="gpa_difference", kde=True, stat="percent", bins=30, ax=axs.ravel()[5])
    axs.ravel()[5].set_title("GPA Difference Distribution")
    
    plt.tight_layout()
    plt.show()


def plot_variables_by_burnout_risk_level(df):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    sns.barplot(
        x=df.groupby(by="burnout_risk_level")["traditional_study_hours"].mean().index,
        y=df.groupby(by="burnout_risk_level")["traditional_study_hours"].mean().values,
        order=["low", "medium", "high"],
        ax=ax1
    )
    sns.pointplot(
        x=df.groupby(by="burnout_risk_level")["traditional_study_hours"].mean().index,
        y=df.groupby(by="burnout_risk_level")["traditional_study_hours"].mean().values,
        order=["low", "medium", "high"],
        ax=ax1,
        color="black"
    )
    ax1.set(
        title="Traditional Study Hours by Burnout Risk Level",
        ylabel="Traditional Study Hours",
        xlabel="Burnout Risk Level"
    )

    sns.barplot(
        x=df.groupby(by="burnout_risk_level")["weekly_genai_hours"].mean().index,
        y=df.groupby(by="burnout_risk_level")["weekly_genai_hours"].mean().values,
        order=["low", "medium", "high"],
        ax=ax2
    )
    sns.pointplot(
        x=df.groupby(by="burnout_risk_level")["weekly_genai_hours"].mean().index,
        y=df.groupby(by="burnout_risk_level")["weekly_genai_hours"].mean().values,
        order=["low", "medium", "high"],
        ax=ax2,
        color="black"
    )
    ax2.set(
        title="Weekly GenAI Hours by Burnout Risk Level",
        ylabel="Weekly GenAI Hours",
        xlabel="Burnout Risk Level"
    )

    sns.barplot(
        x=df.groupby(by="burnout_risk_level")["perceived_ai_dependency"].mean().index,
        y=df.groupby(by="burnout_risk_level")["perceived_ai_dependency"].mean().values,
        order=["low", "medium", "high"],
        ax=ax3
    )
    sns.pointplot(
        x=df.groupby(by="burnout_risk_level")["perceived_ai_dependency"].mean().index,
        y=df.groupby(by="burnout_risk_level")["perceived_ai_dependency"].mean().values,
        order=["low", "medium", "high"],
        ax=ax3,
        color="black"
    )
    ax3.set(
        title="Perceived AI Dependency by Burnout Risk Level",
        ylabel="Perceived AI Dependency",
        xlabel="Burnout Risk Level"
    )

    plt.show()


if __name__ == "__main__":
    data = np.load(file="data/raw/data.npz", allow_pickle=True)

    values = data['data']
    features = data['columns']
    df = pd.DataFrame(
        data=values,
        columns=features
    )

    plot_univariate_distributions(df)