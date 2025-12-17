from _common.configurations import Circle, Rectangle, Configuration
from _common.constants import Water

water_presets = [
    Configuration(
        name="Waterjet",
        information="Water",
        dt=1e-3,
        geometries=[
            Circle(
                material=Water,  # pyright: ignore
                is_continuous=True,
                frame_threshold=i,
                center=(0.5, 0.8),
                velocity=(0, -1),
                radius=0.03,
            )
            for i in range(1, 200)
        ],
    ),
    Configuration(
        name="Waterjet & Pool",
        information="Water",
        dt=1e-3,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0),
                size=(1.0, 0.1),
                velocity=(0, 0),
            ),
            *[
                Circle(
                    material=Water,  # pyright: ignore
                    is_continuous=True,
                    frame_threshold=i,
                    center=(0.5, 0.8),
                    velocity=(0, -1),
                    radius=0.03,
                )
                for i in range(1, 200)
            ],
        ],
    ),
    Configuration(
        name="Dam Break",
        information="Water",
        dt=1e-3,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0),
                size=(0.3, 0.4),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Dam Break, Centered",
        information="Water",
        dt=1e-3,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.35, 0.0),
                size=(0.3, 0.4),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        name="Spherefall, Water",
        information="Water",
        dt=1e-3,
        geometries=[
            Circle(
                material=Water,  # pyright: ignore
                center=(0.5, 0.4),
                velocity=(0, -3),
                radius=0.1,
            ),
        ],
    ),
    Configuration(
        name="Pool",
        information="Water",
        dt=1e-3,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                lower_left=(0.0, 0.0),
                size=(1.0, 0.25),
                velocity=(0, 0),
            ),
        ],
    ),
]
