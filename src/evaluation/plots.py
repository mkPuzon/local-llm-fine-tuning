import matplotlib.pyplot as plt

def plot_loss(trainer, show_val=True):
    history = trainer.state.log_history

    train_steps = []
    train_loss = []
    val_steps = []
    val_loss = []

    for entry in history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])

        if show_val and "eval_loss" in entry and "step" in entry:
            val_steps.append(entry["step"])
            val_loss.append(entry["eval_loss"])

    plt.figure(figsize=(10, 6))

    plt.plot(train_steps, train_loss, label="Training Loss")

    if show_val and val_loss:
        plt.plot(val_steps, val_loss, label="Validation Loss")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss" + (" vs. Validation Loss" if val_loss else ""))
    plt.legend()
    plt.grid(True)
    plt.show()
