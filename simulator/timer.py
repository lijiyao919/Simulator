class Timer:
    _time = 0

    @staticmethod
    def get_time():
        return Timer._time

    @staticmethod
    def tick_time():
        Timer._time += 1

    @staticmethod
    def reset_time():
        Timer._time = 0