from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from fraud_detection_model.config.core import config
from fraud_detection_model.processing import features as pp

fraud_pipe = Pipeline(
    [
        # ========= Date Decomposition ===========#
        # extract date components from transactional date
        ("txn_date_decom", pp.DateDecomp(col=config.model_config.date_transformation)),
        # calculate the daily avg txn amount for credit card and gender
        (
            "avg_txn_amount",
            pp.Aggregator(
                partition_col=config.model_config.avg_day_amt_partition,
                date_col=config.model_config.avg_day_amt_date_cols,
                agg_col=config.model_config.aggregation_col,
                agg_value=config.model_config.sum_aggregation_value,
                agg_type=config.model_config.avg_aggregation_type,
            ),
        ),
        # calculate the yearly std amount per txn by category
        (
            "std_amt_per_txn",
            pp.AggAmtperTxn(
                partition_col=config.model_config.std_day_amt_partition,
                date_col=config.model_config.std_amt_per_txn_year_cols,
                agg_type=config.model_config.std_aggregation_type,
                agg_col=config.model_config.aggregation_col,
            ),
        ),
        # calculate the daily avg amount per txn by credit card
        (
            "avg_amt_per_txn",
            pp.AggAmtperTxn(
                partition_col=config.model_config.amt_per_txn_day_partition,
                date_col=config.model_config.amt_per_txn_date_cols,
                agg_type=config.model_config.avg_aggregation_type,
                agg_col=config.model_config.aggregation_col,
            ),
        ),
        # Aggregate the transaction's hours by fraud proportion
        ("fraud_freq", pp.FraudFreq(col=config.model_config.fraud_freq_mapper)),
        # Hash categories values using Farmhash algoritm
        ("hash_value", pp.FarmHash(col=config.model_config.farm_hash_mapper)),
        # Log transform the numerical variables
        (
            "log_transform",
            LogTransformer(variables=config.model_config.log_transformation),
        ),
        # Normalzied the dataset
        ("normalizer", SklearnTransformerWrapper(transformer=MinMaxScaler())),
        # replace any Nan or null values in the dataset
        ("replace_nan", pp.ReplaceNaN()),
        # select the most influatial variables
        ("selector", pp.SelectFeatures(cols=config.model_config.varaiable_list)),
        # Under defining a under sampling pipeline and adding a decistion treee classifier.
        (
            (
                "sampler_pipe",
                make_pipeline(
                    (RandomUnderSampler(random_state=config.model_config.random_state)),
                    (DecisionTreeClassifier(random_state=config.model_config.seed)),
                ),
            )
        ),
    ]
)
