This is the project brief and I first uploaded this for you to give you the big picture of the project.

## Project Name:

    Modeling the Relationship Between Generative AI Usage and Student Academic Performance.

## Project Scope

    This project aims to demonstrate a complete end-to-end supervised machine learning workflow suitable for a professional Data Science portfolio.

    The emphasis is on:

    - exploratory data analysis
    - feature engineering
    - statistical reasoning
    - reproducible preprocessing
    - modular code organization
    - interpretable regression modeling
    - professional documentation

    The primary objective is not to obtain the absolute best predictive performance, but to demonstrate sound machine learning practices and clear technical communication.

## Important Design Decisions

    Several decisions in this project were intentional.

    Examples include:

    - prioritizing readability and maintainability over excessive optimization
    - using modular Python scripts instead of placing all logic inside notebooks
    - keeping notebooks focused on analysis rather than implementation details
    - preferring interpretable workflows over unnecessarily complex models

    Please evaluate whether these decisions are justified and suggest improvements when appropriate.

## Objective:

    Develop an end-to-end supervised machine learning pipeline that predicts students' post-semester GPA using demographic information, study habits, and AI usage characteristics while extracting meaningful insights through exploratory data analysis and statistical investigation.

## Evaluation Criteria

    When reviewing this project, consider:

    - technical correctness
    - machine learning methodology
    - statistical validity
    - software engineering quality
    - reproducibility
    - documentation quality
    - portfolio value
    - clarity of communication

    Predictive performance is important but should not be the only evaluation criterion.

## Problem Statement:

    The objective is to predict students' post-semester GPA from academic background, study behavior, and AI usage features. Although the dataset contains AI-related variables, the model estimates GPA rather than causal effects of AI.

## Target Variable:

    post_semester_gpa

## Feature list:

    Numerical:
        Student_ID, Pre_Semester_GPA, Weekly_GenAI_Hours, Tool_Diversity, Traditional_Study_Hours, Perceived_AI_Dependency, Skill_Retention_Score

    Categorical:
            Major_Category, Year_of_Study, Primary_Use_Case, Prompt_Engineering_Skill, Paid_Subscription, Institutional_Policy,Anxiety_Level_During_Exams, Burnout_Risk_Level

    Target:
        Post_Semester_GPA

## Dataset Description:

    dataset contains samples and 16 features and we create a new variable named: gpa_difference in it.

    📁 AI_Student_Impact_Dataset:

    🪪 `Identifier:` Student_ID

    🎓 `Academic Profile:` Major_Category, Year_of_Study, Pre/Post GPA

    🤖 `AI Behaviour:` Weekly_GenAI_Hours, Primary_Use_Case, Prompt_Engineering_Skill, Tool_Diversity, Paid_Subscription

    📚 `Study Habits:` Traditional_Study_Hours, Perceived_AI_Dependency

    🏛️ `Institutional:` Institutional_Policy

    🧠 `Well-being:` Anxiety_Level, Skill_Retention_Score, Burnout_Risk_Level
    
    this dataset contains no missing values and the data quality is good.
    number of observations: 50000
    number of predictiors: 15 before encoding
    target variable: post_semester_gpa
    feature types: there are some numerical and some categorical features and a boolean feature.
    source: from Kaggle https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students
    synthetic or real: real
    licence: no licence
    train/test split strategy: I took %20 percent of the data for the test set and the train and evaluation set is about 40000 samples and just at last I evaluate model on the test set to decide that we don't overfit to the data and our model generalizes well. I did cross validation on the training set to get stable evaluation metrics.

## Repository Structure:

    -data:
        -raw:
            -ai_student_impact_dataset (1).csv
            -data.npz
        -processed:
            -processed_data.npz
            - olynomial_features.npz

    -notebooks:
        -eda.ipynb
        -modeling.ipynb
        -statistics.ipynb
    
    -src:
        -feature_engineering.py
        -modeling.py
        -preprocessing.py
        -visualizatioin.py
    
    -utils:
        -eda.py
        -feature_engineering.py
        -modeling.py
        -preprocessing.py
        -visulaization.py
    
    -README.md
    -requirements.txt

    data folder is the place cause maybe we need many versions of the data and we want to test them on different models and this is a placeholder for them for both raw and processed data we have them.
    notebooks folder is needed cause we need a place to write the technical parts of the project and jupyter notebooks are a great environment for experimenting using code and each notebook is for a special task to avoid disorganization and make technical reviews easy.
    src folder is to write the Python scripts specialized for the project to make our notebooks cleaner with less code and write reproducable codes.
    utils folder is for general functions that I will use in every project and will regularly update during the data science projects.
    

## Workflow:

    EDA, Statistics, Model Development, Evaluation.

## Scripts:

    scripts are two types in my project one of them is utils scripts in the utils folder and the other one is src scripts in the src folder. scripts in the utils folder are general functions that I regularly use in differeny projects and their order is: eda.py, feature_engineering.py, modeling.py, preprocessing.py, and visualization.py. scripts in the src folder are for this project to make the codes reusable and to promote modularity in the notebooks and make the code more abstract to make the understanding for the technical viewer better, their order is: feature_engineering.py, modeling.py, preprocesing.py, and visualization.py.

## Current Status:

    The project is considered feature-complete and is currently undergoing a final technical review before publication.

## Known Limitations:

    Statistical analysis may not yet be comprehensive.

    Some methodological choices may benefit from additional validation.

    Further robustness analysis may be appropriate.

## Target users:

    The recruiters, The senior data scientists, ML managers.

## Success criteria:

    Deriving insights and training a regression model with strong generalization, clear interpretation, professional code, good documentation, and poryfolio quality.

## Current Model Performance

    Current best regression model performance:

    - MSE: 0.021
    - RMSE: 0.148
    - MAE: 0.116
    - R²: 0.908

    These results were achieved without extensive hyperparameter optimization. The focus of the project has been developing a clean, reproducible, and interpretable machine learning workflow.

## Planned improvements:

    current:
        Statistical tests
    
    future:
        SHAP analysis (if appropriate)
        robustness analysis
        residual diagnostics
        ablation study
        pipeline improvements

## Questions I want reviewed:

    Technical Review
        Is the problem formulation appropriate?
        Is the preprocessing pipeline correct and reproducible?
        Is feature engineering well justified?
        Is the validation strategy appropriate?
        Is there any target leakage or data leakage?
        Are the evaluation metrics appropriate?
        Are there methodological weaknesses or unsupported assumptions?
        Would you recommend different models? Why?

    Statistical Review
        Which statistical analyses are missing?
        Which statistical tests would meaningfully strengthen the project?
        Are any conclusions statistically unsupported?
        Is the exploratory analysis sufficiently rigorous?

    Software Engineering Review
        Is the repository organized professionally?
        Is the modularization appropriate?
        Is the code maintainable?
        Would this repository be easy for another developer to understand?

    Portfolio Review

        These are the questions I'd especially want Claude to answer:
        Would this project strengthen a Data Scientist portfolio?
        What skills does it demonstrate well?
        What important skills are missing?
        If you were interviewing me based only on this repository, what concerns would you have?
        Does the project appear junior, intermediate, or senior? Explain why.
        Which improvements would provide the greatest return on investment for making the project more competitive?

    Documentation Review
        Is the report structure appropriate?
        Does the README communicate the project effectively?
        What information is missing?
        Is the writing professional enough for recruiters and technical reviewers?

## Expected Deliverables:

    Phase 1:
        Understand every notebook, script, and dataset.
        Produce a concise summary of the project.
        
    Phase 2:
        Conduct a comprehensive technical review.
        Categorize findings as:
        - Essential
        - Recommended
        - Optional