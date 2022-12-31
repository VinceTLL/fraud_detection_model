from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:

    """Check model inputs for unprocessable values."""
    errors = None
    try:

        MultipleFraudDataInputSchema(
            inputs=input_data.replace({np.nan: None}).to_dict(orient="records")
        )

    except ValidationError as error:

        errors = error.json()

    return input_data, errors


class FraudDataInputSchema(BaseModel):

    trans_date_trans_time: Optional[str]
    cc_num: Optional[int]
    merchant: Optional[str]
    category: Optional[str]
    amt: Optional[float]
    first: Optional[str]
    last: Optional[str]
    gender: Optional[str]
    street: Optional[str]
    city: Optional[str]
    state: Optional[str]
    zip: Optional[int]
    lat: Optional[float]
    long: Optional[float]
    city_pop: Optional[int]
    job: Optional[str]
    dob: Optional[str]
    trans_num: Optional[str]
    unix_time: Optional[str]
    merch_lat: Optional[float]
    merch_long: Optional[float]


class MultipleFraudDataInputSchema(BaseModel):
    inputs: List[FraudDataInputSchema]
