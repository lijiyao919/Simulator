import datetime
from simulator.config import *

class Timer:
    _time_step = 0
    _episode = 1

    @staticmethod
    def get_time_step():
        return Timer._time_step

    @staticmethod
    def tick_time_step():
        Timer._time_step += 1

    @staticmethod
    def get_episode():
        return Timer._episode

    @staticmethod
    def tick_episode():
        Timer._episode += 1

    @staticmethod
    def reset_time_step():
        Timer._time_step = 0

    @staticmethod
    def get_time(time):
        return time % TOTAL_MINUTES_ONE_DAY

    @staticmethod
    def get_date(time):
        # have bug in transition from 30/31 to 1st next month, but now just for print out shown
        return (time//TOTAL_MINUTES_ONE_DAY)+DATE

    @staticmethod
    def get_day(time):
        date = (time//TOTAL_MINUTES_ONE_DAY)+DATE
        return datetime.date(YEAR, MONTH, date).isoweekday() if date<=ALL_DAYS_IN_MONTH else datetime.date(YEAR, MONTH+1, 1).isoweekday()

if __name__ == "__main__":
    print("time step: ", Timer.get_time_step())
    Timer.tick_time_step()
    print("time step: ", Timer.get_time_step())
    Timer.reset_time_step()
    print("time step: ", Timer.get_time_step())
    print("time: ", Timer.get_time(2882))
    print("date: ", Timer.get_date(1441))
    print("day: ", Timer.get_day(44640))

