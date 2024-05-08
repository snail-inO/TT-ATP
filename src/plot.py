#!/usr/bin/env python3

import pickle

import matplotlib.pyplot as pt


loss_curves = []
accu_curves = []
norm_curves = []
val_loss_curves = []
val_accu_curves = []
test_loss_curves = []
test_accu_curves = []
for i in range(3):
    with open(f"train_result_{i}.pkl", "rb") as f:
        loss_curve = pickle.load(f)
        accu_curve = pickle.load(f)
        norm_curve = pickle.load(f)
        loss_curves.append(loss_curve)
        accu_curves.append(accu_curve)
        norm_curves.append(norm_curve)

    with open(f"validate_result_{i}.pkl", "rb") as f:
        loss_curve = pickle.load(f)
        accu_curve = pickle.load(f)
        val_loss_curves.append(loss_curve)
        val_accu_curves.append(accu_curve)

    with open(f"test_result_{i}.pkl", "rb") as f:
        loss_curve = pickle.load(f)
        accu_curve = pickle.load(f)
        test_loss_curves.append(loss_curve)
        test_accu_curves.append(accu_curve)


pt.figure(figsize=(15, 5))
pt.suptitle("Training")

# Plot training loss curve
pt.subplot(1, 3, 1)
for i, loss_curve in enumerate(loss_curves):
    pt.plot(loss_curve[::10], label=f'Rep {i+1}')
pt.ylabel("Loss")
pt.xlabel("Per 10 Update")
pt.legend()

# Plot training accuracy curve
pt.subplot(1, 3, 2)
for i, accu_curve in enumerate(accu_curves):
    pt.plot(accu_curve[::20], label=f'Rep {i+1}')
pt.ylabel("Accuracy")
pt.xlabel("Per 20 Update")
pt.legend()

# Plot training gradient norm curve
pt.subplot(1, 3, 3)
for i, norm_curve in enumerate(norm_curves):
    pt.plot(norm_curve[::10], label=f'Rep {i+1}')
pt.ylabel("Gradient Norm")
pt.xlabel("Per 10 Update")
pt.legend()

pt.tight_layout()
pt.savefig("training_curves.png")


pt.figure(figsize=(15, 5))
pt.suptitle("Validation")

# Plot validation loss curve
pt.subplot(1, 2, 1)
for i, val_loss_curve in enumerate(val_loss_curves):
    pt.plot(val_loss_curve, label=f'Rep {i+1}')
pt.ylabel("Loss")
pt.xlabel("Batch")
pt.legend()

# Plot validation accuracy curve
pt.subplot(1, 2, 2)
for i, val_accu_curve in enumerate(val_accu_curves):
    pt.plot(val_accu_curve, label=f'Rep {i+1}')
pt.ylabel("Accuracy")
pt.xlabel("Batch")
pt.legend()

pt.tight_layout()
pt.savefig("validation_curves.png")


pt.figure(figsize=(15, 5))
pt.suptitle("Test")

# Plot test loss curve
pt.subplot(1, 2, 1)
for i, test_loss_curve in enumerate(test_loss_curves):
    pt.plot(test_loss_curve, label=f'Rep {i+1}')
pt.ylabel("Loss")
pt.xlabel("Batch")
pt.legend()

# Plot test accuracy curve
pt.subplot(1, 2, 2)
for i, test_accu_curve in enumerate(test_accu_curves):
    pt.plot(test_accu_curve, label=f'Rep {i+1}')
pt.ylabel("Accuracy")
pt.xlabel("Batch")
pt.legend()

pt.tight_layout()
pt.savefig("test_curves.png")