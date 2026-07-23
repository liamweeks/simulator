from dataclasses import dataclass
from typing import Optional


@dataclass
class Packet:
    id: int
    arrival: float
    service_start: Optional[float] = None
    departure: Optional[float] = None


    def __str__(self):
        return f"Packet {self.id}: {self.arrival}->{self.service_start}->{self.departure}"