from sys import argv

from simulator import RaceTrack, Simulator, plt
from controller import init_controller

if __name__ == "__main__":
    assert(len(argv) == 3)
    racetrack = RaceTrack(argv[1])
    raceline_path = argv[2]

    init_controller(raceline_path)

    simulator = Simulator(racetrack)
    simulator.start()
    plt.show()