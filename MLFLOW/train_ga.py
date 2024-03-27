import mlflow
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from mlflow.models.signature import infer_signature

if __name__ == "__main__":
    ### MLFLOW Experiment setup
    experiment_name = "getaround_experiment"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)

    # Call mlflow autolog
    mlflow.sklearn.autolog(log_models=False)

    # Carga de datos
    df_pricing = pd.read_csv("get_around_pricing_project.csv")

    # Definición de características
    all_features = ['model_key', 'mileage', 'engine_power', 'private_parking_available', 'has_gps', 'fuel', 'paint_color', 'car_type',
                    'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
    numeric_features = ['mileage', 'engine_power']
    categorical_features = ['model_key', 'fuel', 'paint_color', 'car_type', 'private_parking_available', 'has_gps',
                            'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires']
    target_feature = 'rental_price_per_day'

    # División de datos en conjunto de entrenamiento y prueba
    X_train, X_test, Y_train, Y_test = train_test_split(df_pricing[all_features], df_pricing[target_feature], test_size=0.2, random_state=5)

    # Preprocesamiento de datos
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(drop='first'))])  

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Entrenamiento y evaluación del modelo de Regresión Lineal
    with mlflow.start_run(run_id=run.info.run_id):
        model_lr = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
        model_lr.fit(X_train, Y_train)

        Y_train_pred_lr = model_lr.predict(X_train)
        Y_test_pred_lr = model_lr.predict(X_test)

        # Registro del modelo y métricas en MLflow
        mlflow.sklearn.log_model(
            sk_model=model_lr,
            artifact_path="getaround_project_lr_model",
            signature=infer_signature(X_train, Y_train_pred_lr)
        )

        mlflow.log_metric("training_r2_score", r2_score(Y_train, Y_train_pred_lr))
        mlflow.log_metric("training_mean_absolute_error", mean_absolute_error(Y_train, Y_train_pred_lr))
        mlflow.log_metric("training_mean_squared_error", mean_squared_error(Y_train, Y_train_pred_lr))
        mlflow.log_metric("training_root_mean_squared_error", mean_squared_error(Y_train, Y_train_pred_lr, squared=False))
        mlflow.log_metric("testing_r2_score", r2_score(Y_test, Y_test_pred_lr))
        mlflow.log_metric("testing_mean_absolute_error", mean_absolute_error(Y_test, Y_test_pred_lr))
        mlflow.log_metric("testing_mean_squared_error", mean_squared_error(Y_test, Y_test_pred_lr))
        mlflow.log_metric("testing_root_mean_squared_error", mean_squared_error(Y_test, Y_test_pred_lr, squared=False))
        mlflow.log_metric("training_r2_score", r2_score(Y_train, Y_train_pred_lr))
        mlflow.log_metric("training_standard_deviation", np.std(Y_train_pred_lr))
        
        mlflow.log_metric("testing_standard_deviation", np.std(Y_test_pred_lr))

    # Entrenamiento y evaluación del modelo de Árbol de Decisión
    run_name = 'decision_tree'
    with mlflow.start_run(run_name=run_name) as run_dt:
        model_dt = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', DecisionTreeRegressor(max_depth=10, random_state=42))])
        model_dt.fit(X_train, Y_train)
        
        Y_train_pred_dt = model_dt.predict(X_train)
        Y_test_pred_dt = model_dt.predict(X_test)
        
        # Registro del modelo y métricas en MLflow para el árbol de decisión
        mlflow.sklearn.log_model(
            sk_model=model_dt,
            artifact_path="getaround_project_dt_model",
            signature=infer_signature(X_train, Y_train_pred_dt)
        )
        
        mlflow.log_metric("training_r2_score", r2_score(Y_train, Y_train_pred_dt))
        mlflow.log_metric("training_mean_absolute_error", mean_absolute_error(Y_train, Y_train_pred_dt))
        mlflow.log_metric("training_mean_squared_error", mean_squared_error(Y_train, Y_train_pred_dt))
        mlflow.log_metric("training_root_mean_squared_error", mean_squared_error(Y_train, Y_train_pred_dt, squared=False))
        mlflow.log_metric("testing_r2_score", r2_score(Y_test, Y_test_pred_dt))
        mlflow.log_metric("testing_mean_absolute_error", mean_absolute_error(Y_test, Y_test_pred_dt))
        mlflow.log_metric("testing_mean_squared_error", mean_squared_error(Y_test, Y_test_pred_dt))
        mlflow.log_metric("testing_root_mean_squared_error", mean_squared_error(Y_test, Y_test_pred_dt, squared=False))