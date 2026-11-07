from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True,slots=True)
class Advance:
    target:int

@dataclass(frozen=True,slots=True)
class Patch:
    machine:int

@dataclass(frozen=True,slots=True)
class Firewall:
    a:int
    b:int
    
    def __post_init__(self)->None:
        if self.a>=self.b:
            raise ValueError("write the smaller number first")
    

@dataclass(frozen=True,slots=True)
class Honeypot:
    machine:int

@dataclass(frozen=True,slots=True)
class Pass:
    pass

Action:TypeAlias= Advance | Patch | Firewall | Honeypot | Pass 

