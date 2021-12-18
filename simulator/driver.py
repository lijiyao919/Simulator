class Driver:
    IDLE = "idle"
    BUSY = "busy"

    def __init__(self, wID, pos):
        self._id = wID
        self._pos = pos

        self._status = Driver.IDLE
        self._rider_id = None
        self._total_relocate_effort = 0
        self._total_trip_effort = 0
