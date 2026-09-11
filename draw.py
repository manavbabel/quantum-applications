import matplotlib.pyplot as plt
from matplotlib import patches


def draw(problem, start_times, title="Task Scheduling Gantt Chart"):
    # problem tasks might have a trailing zero-duration makespan task which should have no bar
    # as a result start_times might be longer than tasks
    tasks = problem.tasks
    start_times = list(start_times)[: len(tasks)]

    bars = [
        (i, machine, duration)
        for i, (machine, duration, _) in enumerate(tasks)
        if duration
    ]
    num_machines = problem.num_machines
    total = problem.makespan(start_times)

    figure, ax = plt.subplots(figsize=(10, 6))
    for i, machine, duration in bars:
        ax.add_patch(
            patches.Rectangle(
                (start_times[i], machine - 0.5),
                duration,
                1,
                edgecolor="black",
                facecolor="skyblue",
            )
        )
        ax.text(
            start_times[i] + duration / 2,
            machine,
            f"Task {i}",
            ha="center",
            va="center",
        )

    ax.set_xlim(0, total)
    ax.set_ylim(-1, num_machines)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i}" for i in range(num_machines)])
    ax.grid(True)
    ax.set_title(f"{title} (makespan {total})")
    figure.tight_layout()
    plt.show()
    