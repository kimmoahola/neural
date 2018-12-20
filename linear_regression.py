import os

import numpy as np
import tensorflow as tf

import imports85    # pylint: disable=g-bad-import-order

STEPS = 300


def main(argv):
    (train, test) = imports85.dataset()

    # Build the training input_fn.
    def input_train():
        return (
                # Shuffling with a buffer larger than the data set ensures
                # that the examples are well mixed.
                train.shuffle(1000).batch(128)
                # Repeat forever
                .repeat().make_one_shot_iterator().get_next())

    # Build the validation input_fn.
    def input_test():
        return test.shuffle(1000).batch(128).make_one_shot_iterator().get_next()

    feature_columns = [
        tf.feature_column.numeric_column(key="target"),
        tf.feature_column.numeric_column(key="error"),
        tf.feature_column.numeric_column(key="error-rate"),
        tf.feature_column.numeric_column(key="outside"),
        tf.feature_column.numeric_column(key="outside-12h"),
        tf.feature_column.numeric_column(key="i-term"),
        tf.feature_column.numeric_column(key="last-was-heating"),
    ]

    # Build the Estimator.
    model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    # Train the model.
    # By default, the Estimators log output every 100 steps.
    model.train(input_fn=input_train, steps=STEPS)

    # Evaluate how the model performs on data it has not yet seen.
    eval_result = model.evaluate(input_fn=input_test)

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    # Convert MSE to Root Mean Square Error (RMSE).
    print("\n" + 80 * "*")
    print("\nMS  error for the test set: {:.2f}".format(average_loss))
    print("\nRMS error for the test set: {:.2f}".format(average_loss**0.5))

    # Run the model in prediction mode.
    input_dict = {
            "target": np.array([3, 6, 5, 5]),
            "error": np.array([-1, 1, 0, 0.2]),
            "error-rate": np.array([-0.1, 0.1, 0.2, 0]),
            "outside": np.array([-5, -5, 5, -4]),
            "outside-12h": np.array([-7, -2, 2, -2]),
            "i-term": np.array([0, 0, 0, 0.1]),
            "last-was-heating": np.array([1, 1, 1, 0]),
    }
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(input_dict, shuffle=False)
    predict_results = model.predict(input_fn=predict_input_fn)

    # Print the prediction results.
    print("\nPrediction results:")
    for i, prediction in enumerate(predict_results):
        msg = "target: {: 4.1f}, error: {: 0.1f}, error-rate: {: 0.1f}, outside: {: 0.1f}, outside-12h: {: 0.1f}, " \
              "i-term: {: 0.1f}, last-was-heating: {: 0.1f}, " \
              "Should heat prediction: {: 9.2f}"
        msg = msg.format(
            input_dict["target"][i],
            input_dict["error"][i],
            input_dict["error-rate"][i],
            input_dict["outside"][i],
            input_dict["outside-12h"][i],
            input_dict["i-term"][i],
            input_dict["last-was-heating"][i],
            prediction["predictions"][0]
        )

        print("        " + msg)

    print()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf.logging.set_verbosity(tf.logging.ERROR)

    # The Estimator periodically generates "INFO" logs; make these logs visible.
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
