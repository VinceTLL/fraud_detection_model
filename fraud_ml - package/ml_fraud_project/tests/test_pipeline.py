import math
from datetime import datetime

from sklearn.metrics import recall_score

from fraud_detection_model.config.core import config
from fraud_detection_model.predict import make_prediction
from fraud_detection_model.processing import features as pp


def test_date_transformer(sample_input_train_data, sample_input_test_data):

    # gven
    date_transformer = pp.DateDecomp(col=config.model_config.date_transformation)

    subject = date_transformer.fit_transform(sample_input_train_data)

    # when

    for index, (date, dob) in enumerate(
        zip(
            sample_input_train_data["trans_date_trans_time"].tolist(),
            sample_input_train_data["dob"].tolist(),
        )
    ):

        assert sample_input_train_data["trans_date_trans_time"].iat[index] == date
        assert sample_input_train_data["dob"].iat[index] == dob

        date_list = (
            sample_input_train_data["trans_date_trans_time"]
            .iat[index]
            .split()[0]
            .split("-")
        )
        time_list = (
            sample_input_train_data["trans_date_trans_time"]
            .iat[index]
            .split()[1]
            .split(":")
        )
        date_str = (
            sample_input_train_data["trans_date_trans_time"].iat[index].split()[0]
        )

        dob_list = sample_input_train_data["dob"].iat[index].split("-")

        year_dob, month_dob, day_dob = (
            int(dob_list[0]),
            int(dob_list[1]),
            int(dob_list[2]),
        )
        year, month, day = int(date_list[0]), int(date_list[1]), int(date_list[2])
        hour = int(time_list[0])

        # then
        # testing the trans_date_trans_time column
        assert subject["trans_date_trans_time_year"].iat[index] == year
        assert subject["trans_date_trans_time_month"].iat[index] == month
        assert subject["trans_date_trans_time_day"].iat[index] == day
        assert (
            datetime.strftime(
                subject["trans_date_trans_time_caldate"].iat[index], format="%Y-%m-%d"
            )
            == date_str
        )
        assert subject["trans_date_trans_time_hour"].iat[index] == hour

        # testing the dob column
        assert subject["dob_year"].iat[index] == year_dob
        assert subject["dob_month"].iat[index] == month_dob
        assert subject["dob_day"].iat[index] == day_dob


def test_aggregator_transformed(sample_input_train_data, sample_input_test_data):

    # given
    aggregator = pp.Aggregator(
        partition_col=config.model_config.avg_day_amt_partition,
        date_col=config.model_config.avg_day_amt_date_cols,
        agg_col=config.model_config.aggregation_col,
        agg_value=config.model_config.sum_aggregation_value,
        agg_type=config.model_config.avg_aggregation_type,
    )

    date_transformer = pp.DateDecomp(col=config.model_config.date_transformation)

    subject_date_transformed = date_transformer.fit_transform(sample_input_train_data)

    subject = aggregator.fit_transform(subject_date_transformed)

    assert subject["cc_num"].iat[0] == 2703186189652095
    assert subject["cc_num"].iat[4] == 375534208663984

    # then
    assert math.isclose(
        subject["cc_num_mean_caldate_sum_amt"].iat[0], 345.484288, abs_tol=0.001
    )
    assert math.isclose(
        subject["cc_num_mean_caldate_sum_amt"].iat[4], 372.765456, abs_tol=0.001
    )


def test_aggamtpertxn(sample_input_train_data, sample_input_test_data):

    agg_per_txn_amt = pp.AggAmtperTxn(
        partition_col=config.model_config.amt_per_txn_day_partition,
        date_col=config.model_config.amt_per_txn_date_cols,
        agg_type=config.model_config.avg_aggregation_type,
        agg_col=config.model_config.aggregation_col,
    )

    date_transformer = pp.DateDecomp(col=config.model_config.date_transformation)
    subject_date_transformed = date_transformer.fit_transform(sample_input_train_data)
    subject = agg_per_txn_amt.fit_transform(subject_date_transformed)

    assert subject["cc_num"].iat[0] == 2703186189652095
    assert subject["cc_num"].iat[4] == 375534208663984
    # then
    assert math.isclose(
        subject["cc_num_mean_caldate_amt_per_txn"].iat[0], 85.442088, abs_tol=0.001
    )
    assert math.isclose(
        subject["cc_num_mean_caldate_amt_per_txn"].iat[4], 94.427516, abs_tol=0.001
    )


def test_fraud_freq(sample_input_train_data, sample_input_test_data):

    fraud_freq = pp.FraudFreq(col=config.model_config.fraud_freq_mapper)

    date_transformer = pp.DateDecomp(col=config.model_config.date_transformation)
    subject_date_transformed = date_transformer.fit_transform(sample_input_train_data)
    target = sample_input_train_data[config.model_config.target]

    subject = fraud_freq.fit_transform(subject_date_transformed, target)

    assert subject["trans_date_trans_time_hour"].iat[0] == 0
    assert subject["trans_date_trans_time_hour"].iat[1296670] == 12
    # then

    assert math.isclose(
        subject["trans_date_trans_time_hour_fraud_freq"].iat[0],
        0.01494,
        abs_tol=0.00001,
    )
    assert math.isclose(
        subject["trans_date_trans_time_hour_fraud_freq"].iat[1296670],
        0.001027,
        abs_tol=0.00001,
    )


def test_recall_score(sample_input_train_data, sample_input_test_data):

    truth = sample_input_test_data[config.model_config.target]

    predictions = make_prediction(
        input_data=sample_input_test_data[config.model_config.features]
    )

    score = recall_score(truth, predictions["predictions"])

    assert score == 0.9631701631701631
