import pandas as pd
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    # creating the inputs 
    feature_names = [feature for feature in df.columns if feature not in ["student_id", "gpa_difference", "post_semester_gpa"]]
    X = df[feature_names]

    # encoding
    oh_encoding_cols = ["major_category", "primary_use_case", "institutional_policy"]
    ordinal_cols = ["year_of_study", "prompt_engineering_skill", "burnout_risk_level"]
    mapping = [
        {
            "col": "year_of_study",
            "mapping": {
                "freshman": 0,
                "sophomore": 1,
                "junior": 2,
                "senior": 3,
                "graduate": 4
            }
        },
        {
            "col": "prompt_engineering_skill",
            "mapping":{
                "beginner": 0,
                "intermediate": 1,
                "advanced": 2
            }
        },
        {
            "col": "burnout_risk_level",
            "mapping": {
                "low": 0,
                "medium": 1,
                "high": 2
            }
        }
    ]

    encoder = Pipeline(
        steps=[
            ("one_hot", ce.OneHotEncoder(cols=oh_encoding_cols, use_cat_names=True)),
            ("ordinal_1", ce.OrdinalEncoder(mapping=mapping, cols=ordinal_cols))
        ]
    )
    features_encoded = pd.DataFrame(
        data=encoder.fit_transform(X),
        columns=encoder.get_feature_names_out()
    )

    # droping the non-informative column in the one-hot encoded features (one is non-informative in each)
    non_informative_features = ["major_category_arts", "primary_use_case_direct_answer_generation", "institutional_policy_actively_encouraged"]
    informative_features = [feature for feature in features_encoded.columns if feature not in non_informative_features]
    features_encoded = features_encoded[informative_features]
    return features_encoded


def scale_data(X_train, X_test):
    scaler = ColumnTransformer(
        transformers=[("scaler", StandardScaler(), [5, 13, 18])]
    )
    scaled_train_columns = scaler.fit_transform(X_train)
    scaled_test_columns = scaler.transform(X_test)

    X_train_scaled = X_train
    X_train_scaled[:, 5] = scaled_train_columns[:, 0]
    X_train_scaled[:, 13] = scaled_train_columns[:, 1]
    X_train_scaled[:, 18] = scaled_train_columns[:, 2]
    X_test_scaled = X_test
    X_test_scaled[:, 5] = scaled_test_columns[:, 0]
    X_test_scaled[:, 13] = scaled_test_columns[:, 1]
    X_test_scaled[:, 18] = scaled_test_columns[:, 2]

    return X_train_scaled, X_test_scaled
