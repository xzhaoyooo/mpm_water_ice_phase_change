from dataclasses import dataclass
from taichi import hex_to_rgb


@dataclass
class Classification:
    Empty = 22
    Colliding = 33
    Interior = 44
    Insulated = 55


@dataclass
class ColorHEX:
    HeatMap = [0x323296, 0x5050AB, 0x7575BF, 0xDB7F85, 0xD64F58, 0xC73C45]
    Background = 0x007D79  # teal 60
    Water = 0x78A9FF  # blue 40
    Ice = 0xD0E2FF  # blue 20


@dataclass
class ColorRGB:
    HeatMap = [hex_to_rgb(color) for color in ColorHEX.HeatMap]
    Background = hex_to_rgb(ColorHEX.Background)
    Water = hex_to_rgb(ColorHEX.Water)
    Ice = hex_to_rgb(ColorHEX.Ice)


@dataclass
class State:
    Active = 0
    Hidden = 1


@dataclass
class Simulation:
    """Defines parameters for the simulation."""

    MininumTemperature = -273.15
    MaximumTemperature = 100.0


@dataclass
class Material:
    """Defines parameters that represent a material."""

    Conductivity: float = 0.0
    LatentHeat: float = 0.0
    Capacity: float = 0.0
    Density: float = 0.0
    Lambda: float = 0.0
    Color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    Phase: int = 0
    Mu: float = 0.0


@dataclass
class Water(Material):
    """Defines parameters that represent water."""

    Conductivity = 0.55
    LatentHeat = 334.4
    Capacity = 4.186  # j/dC
    Density = 997.0
    Lambda = 1e32
    Color = ColorRGB.Water
    Phase = 43
    Mu = 0


E = 2.8e5
nu = 0.15


@dataclass
class Ice(Material):
    """Defines parameters that represent ice."""

    Conductivity = 2.33
    LatentHeat = 0.0
    Capacity = 2.093  # j/dC
    Density = 400.0
    Lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    Color = ColorRGB.Ice
    Phase = 14
    Mu = E / (2 * (1 + nu))
