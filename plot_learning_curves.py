import matplotlib.pyplot as plt

# ====== DATA (من اللوج تبعك) ======
epochs = list(range(1, 21))

train_loss = [
    0.9039, 0.7853, 0.5998, 0.4171, 0.2941,
    0.2327, 0.1793, 0.1682, 0.1515, 0.1541,
    0.1463, 0.1542, 0.1504, 0.1644, 0.1176,
    0.1524, 0.1832, 0.1362, 0.1239, 0.1177
]

val_loss = [
    1.0102, 1.0604, 1.1669, 1.2826, 1.4771,
    1.7313, 2.2751, 2.6634, 2.6631, 1.2995,
    0.2043, 0.0738, 0.0625, 0.0553, 0.0534,
    0.0514, 0.0512, 0.0536, 0.0553, 0.0452
]

train_f1 = [
    0.5143, 0.6352, 0.7423, 0.8357, 0.8934,
    0.9082, 0.9346, 0.9370, 0.9362, 0.9357,
    0.9459, 0.9434, 0.9409, 0.9351, 0.9474,
    0.9428, 0.9296, 0.9410, 0.9529, 0.9492
]

val_f1 = [
    0.4696, 0.4696, 0.4855, 0.4144, 0.4144,
    0.4144, 0.4144, 0.4144, 0.4199, 0.4975,
    0.9151, 0.9605, 0.9552, 0.9657, 0.9637,
    0.9693, 0.9776, 0.9710, 0.9728, 0.9829
]

train_acc_elem = [
    0.5449, 0.6906, 0.7945, 0.8820, 0.9247,
    0.9357, 0.9555, 0.9573, 0.9573, 0.9569,
    0.9635, 0.9608, 0.9608, 0.9555, 0.9648,
    0.9639, 0.9520, 0.9613, 0.9688, 0.9652
]

val_acc_elem = [
    0.5506, 0.5506, 0.4405, 0.5139, 0.5139,
    0.5139, 0.5139, 0.5139, 0.5243, 0.6195,
    0.9430, 0.9722, 0.9683, 0.9747, 0.9752,
    0.9787, 0.9826, 0.9787, 0.9792, 0.9861
]

train_acc_subset = [
    0.0387, 0.1461, 0.4630, 0.7183, 0.7975,
    0.8187, 0.8644, 0.8644, 0.8732, 0.8697,
    0.8908, 0.8856, 0.8856, 0.8697, 0.8944,
    0.8891, 0.8574, 0.8803, 0.9014, 0.8979
]

val_acc_subset = [
    0.0000, 0.0000, 0.0000, 0.1825, 0.1825,
    0.1825, 0.1825, 0.1825, 0.1825, 0.2798,
    0.8135, 0.9167, 0.9008, 0.9127, 0.9286,
    0.9286, 0.9583, 0.9425, 0.9583, 0.9722
]


def _plot_curve(x, y1, y2, y1_label, y2_label, title, y_label, out_png):
    plt.figure()
    plt.plot(x, y1, label=y1_label)
    plt.plot(x, y2, label=y2_label)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()


def main():
    # 1) LOSS
    _plot_curve(
        epochs, train_loss, val_loss,
        "Train Loss", "Validation Loss",
        "Learning Curve - Loss", "Loss",
        "learning_curve_loss.png"
    )

    # 2) F1
    _plot_curve(
        epochs, train_f1, val_f1,
        "Train F1", "Validation F1",
        "Learning Curve - F1 Score", "F1 Score",
        "learning_curve_f1.png"
    )

    # 3) ELEMENT ACCURACY
    _plot_curve(
        epochs, train_acc_elem, val_acc_elem,
        "Train Element Accuracy", "Validation Element Accuracy",
        "Learning Curve - Element Accuracy", "Accuracy",
        "learning_curve_element_accuracy.png"
    )

    # 4) SUBSET ACCURACY
    _plot_curve(
        epochs, train_acc_subset, val_acc_subset,
        "Train Subset Accuracy", "Validation Subset Accuracy",
        "Learning Curve - Subset Accuracy", "Accuracy",
        "learning_curve_subset_accuracy.png"
    )

    print("\n✅ Saved plots:")
    print(" - learning_curve_loss.png")
    print(" - learning_curve_f1.png")
    print(" - learning_curve_element_accuracy.png")
    print(" - learning_curve_subset_accuracy.png")


if __name__ == "__main__":
    main()
