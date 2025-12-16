from _common.configurations import Circle, Configuration, Rectangle
from _common.constants import Ice, Snow, PurpleSnow, MagentaSnow

# Width of the bounding box, TODO: transform points to coordinates in bounding box
configuration_list = [
    Configuration(
        name="Snowball hits wall [1]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.5, 0.5),
                velocity=(4, 0),
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits ground [1]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.5, 0.5),
                velocity=(0, -3),
                radius=0.08,
            ),
        ],
    ),
    Configuration(
        name="Ice Cube Fall",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.425, 0.425),
                temperature=-10.0,
            )
        ],
    ),
    Configuration(
        name="Ice Cube vs. Snow Cube",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.225, 0.425),
                temperature=-10.0,
            ),
            Rectangle(
                material=Snow,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.625, 0.425),
                temperature=-10.0,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball [1]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.07, 0.595),
                velocity=(3, 0),
                radius=0.04,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.91, 0.615),
                velocity=(-3, 0),
                radius=0.06,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (colored) [1]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=MagentaSnow,  # pyright: ignore
                center=(0.07, 0.595),
                velocity=(3, 0),
                radius=0.04,
            ),
            Circle(
                material=PurpleSnow,  # pyright: ignore
                center=(0.91, 0.615),
                velocity=(-3, 0),
                radius=0.06,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball [2]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.1, 0.5),
                velocity=(4, 0),
                radius=0.07,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.9, 0.53),
                velocity=(-8, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (colored) [2]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=PurpleSnow,  # pyright: ignore
                center=(0.1, 0.5),
                velocity=(4, 0),
                radius=0.07,
            ),
            Circle(
                material=MagentaSnow,  # pyright: ignore
                center=(0.9, 0.53),
                velocity=(-8, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (high velocity) [3]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.08, 0.5),
                velocity=(7, 0),
                radius=0.07,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.9, 0.51),
                velocity=(-7, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits snowball (colored, high velocity) [3]",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=MagentaSnow,  # pyright: ignore
                center=(0.08, 0.5),
                velocity=(7, 0),
                radius=0.07,
            ),
            Circle(
                material=PurpleSnow,  # pyright: ignore
                center=(0.9, 0.51),
                velocity=(-7, 0),
                radius=0.07,
            ),
        ],
    ),
    Configuration(
        name="Snowball hits giant snowball",
        dt = 1e-4,
        gravity = -9.81,
        geometries=[
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.08, 0.5),
                velocity=(10, 0),
                radius=0.05,
            ),
            Circle(
                material=Snow,  # pyright: ignore
                center=(0.79, 0.51),
                velocity=(-1, 0),
                radius=0.15,
            ),
        ],
    ),
]

# Sort by length in descending order:
# TODO: move sorting an stuff to BaseSimuluation or something
# configuration_list.sort(key=lambda c: len(c.name), reverse=True)
