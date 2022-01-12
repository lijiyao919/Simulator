import datetime
from simulator.config import *

class Timer:
    _time_step = 0

    @staticmethod
    def get_time_step():
        return Timer._time_step

    @staticmethod
    def tick_time_step():
        Timer._time_step += 1

    @staticmethod
    def reset_time_step():
        Timer._time_step = 0

    @staticmethod
    def get_time(time):
        return time % TOTAL_MINUTES_ONE_DAY

    @staticmethod
    def get_day(time):
        date = (time//TOTAL_MINUTES_ONE_DAY)+1
        return datetime.date(YEAR, MONTH, date).isoweekday()

if __name__ == "__main__":
    print("time step: ", Timer.get_time_step())
    Timer.tick_time_step()
    print("time step: ", Timer.get_time_step())
    Timer.reset_time_step()
    print("time step: ", Timer.get_time_step())
    print("time: ", Timer.get_time(2882))
    print("day: ", Timer.get_day(2882))

