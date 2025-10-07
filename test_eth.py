"""
Etheria test for XOR problem
"""
import etheria as eth
import numpy as np


def main():
    model = eth.SequentialModel(
        input_shape=(2,),
        layers=[
            eth.DenseLayer(neurons=4, activation='relu'),
            eth.DenseLayer(neurons=1, activation='sigmoid')
        ],
        dtype=eth.DType.FLOAT32
    )

    print(model.summary())

    images = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float32)
    labels = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float32)

    model.configure(
        learning_rate=0.1,
        optimizer='sgd', 
        loss='mse'
    )

    model.train(images, labels, epochs=50, verbose=True)
    
    predictions = model.predict(images)
    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print(f"Input: {images[i]}, Predicted: {prediction}, Actual: {labels[i]}")

    stats = [
        'loss',
        'accuracy'
    ]

    results = model.evaluate(images, labels, stats=stats)
    
    print("Evaluation Results:")
    for stat, value in results.items():
        print(f"{stat}: {value}")

if __name__ == "__main__":
    main()