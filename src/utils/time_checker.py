import time


class TimeChecker:
    def __init__(self, program_name: str) -> None:
        self.program_name = program_name

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        self.time_comsunption = self.end_time - self.start_time
        print(f"{self.program_name}: {self.time_comsunption}")
