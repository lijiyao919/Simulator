import datetime

TOTAL_MINUTES_ONE_DAY = 1440
YEAR = 2019
MONTH = 8

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
    def get_time():
        return Timer._time_step % TOTAL_MINUTES_ONE_DAY

    @staticmethod
    def get_day():
        date = (Timer._time_step//TOTAL_MINUTES_ONE_DAY)+1
        return datetime.date(YEAR, MONTH, date).isoweekday()

if __name__ == "__main__":
    Timer._time_step = 2881
    print("time step: ", Timer.get_time_step())
    Timer.tick_time_step()
    print("time step: ", Timer.get_time_step())
    print("time: ", Timer.get_time())
    print("day: ", Timer.get_day())

