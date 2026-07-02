from fraudrisk_engine.data import FEATURE_COLUMNS, TARGET_COLUMN, generate_transactions


def test_generate_transactions_has_expected_schema_and_fraud_rate() -> None:
    df = generate_transactions(n_rows=1000, seed=7)

    assert len(df) == 1000
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]).issubset(df.columns)
    assert df["transaction_id"].is_unique
    assert 0.02 <= df[TARGET_COLUMN].mean() <= 0.30
