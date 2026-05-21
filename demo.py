from random import seed

from MLP import MLP
from node import Value


def show_forward_pass():
    seed(42)

    model = MLP(2, [4, 4, 1])
    sample = [2.0, -1.0]

    output = model(sample)
    print("Forward pass demo")
    print(f"  input:  {sample}")
    print(f"  output: {output}")


def train_xor_style_demo():
    seed(42)

    model = MLP(2, [4, 4, 1])
    data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    learning_rate = 0.05

    print("\nTiny training demo")
    for epoch in range(100):
        loss = Value(0)

        for x, y in data:
            prediction = model(x)
            assert isinstance(prediction, Value)
            diff = (Value(y) * -1) + prediction
            loss = loss + diff * diff

        model.zero_grad()
        loss.backward()

        for param in model.parameters():
            param.data += -learning_rate * param.grad

        print(f"  epoch {epoch + 1:02d}: loss={loss.data:.6f}")

    print("\nPredictions after training")
    for x, y in data:
        prediction = model(x)
        assert isinstance(prediction, Value) # sanity check that the output is a single Value
        print(f"  input={x} target={y:.1f} prediction={prediction.data:0.6f}")


def main():
    show_forward_pass()
    train_xor_style_demo()


if __name__ == "__main__":
    main()
