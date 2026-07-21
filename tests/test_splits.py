import numpy as np

from data.splits import create_stratified_split, validate_split_indices


def test_stratified_split_is_reproducible_and_disjoint():
    labels = np.repeat(np.arange(10), 60)
    train_a, validation_a = create_stratified_split(
        labels, validation_size=100, seed=0
    )
    train_b, validation_b = create_stratified_split(
        labels, validation_size=100, seed=0
    )
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    validate_split_indices(
        train_a,
        validation_a,
        dataset_size=600,
        expected_validation_size=100,
    )
    validation_counts = np.bincount(labels[validation_a], minlength=10)
    np.testing.assert_array_equal(validation_counts, np.full(10, 10))
