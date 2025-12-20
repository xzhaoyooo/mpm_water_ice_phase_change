from _common.configurations import Circle, Rectangle, Configuration
from _common.constants import Ice, Water

mixed_presets = [
    Configuration(
        name="Melting Ice Cube, Floating",
        information="Ice -> Water",
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.15, 0.15),
                velocity=(0, 0),
                lower_left=(0.425, 0.425),
                temperature=-10.0,
            )
        ],
        ambient_temperature=100.0,
        gravity=0.0,
    ),
    Configuration(
        name="Melting Ice Ball, Floating",
        dt=1e-4,
        information="Ice -> Water",
        geometries=[
            Circle(
                velocity=(0, 0),
                center=(0.5, 0.5),
                radius=0.1,
                temperature=-10.0,
                material=Ice,  # pyright: ignore
            )
        ],
        ambient_temperature=100.0,
        gravity=0.0,
    ),
    Configuration(
        name="Melting Ice Ball",
        dt=1e-4,
        information="Ice -> Water",
        geometries=[
            Circle(
                velocity=(0, 0),
                center=(0.5, 0.1),
                radius=0.1,
                temperature=-10.0,
                material=Ice,  # pyright: ignore
            )
        ],
        ambient_temperature=100.0,
        gravity=0.0,
    ),
    Configuration(
        dt=1e-4,
        name="Melting Ice Cube",
        information="Ice -> Water",
        ambient_temperature=100.0,
        geometries=[
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.2, 0.21),  # a bit higher to fit the grid better
                velocity=(0, 0),
                lower_left=(0.4, 0.0),
                temperature=-10.0,
            ),
        ],
    ),
    Configuration(
        dt=1e-4,
        name="Pool & Ice Cubes",
        information="Water, Ice -> Water",
        ambient_temperature=30.0,
        geometries=[
            Rectangle(
                material=Water,  # pyright: ignore
                size=(1.0, 0.1),
                velocity=(0, 0),
                lower_left=(0, 0),
                frame_threshold=0,
                temperature=50.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.08, 0.08),
                velocity=(0, -2),
                lower_left=(0.25, 0.35),
                frame_threshold=10,
                temperature=-10.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.08, 0.08),
                velocity=(0, -2),
                lower_left=(0.45, 0.15),
                frame_threshold=70,
                temperature=-10.0,
            ),
            Rectangle(
                material=Ice,  # pyright: ignore
                size=(0.08, 0.08),
                velocity=(0, -2),
                lower_left=(0.65, 0.25),
                frame_threshold=130,
                temperature=-10.0,
            ),
        ],
    ),
    Configuration(
        dt=1e-3,
        name="Freezing Pool",
        information="Water -> Ice",
        ambient_temperature=-500.0,
        geometries=[
            Rectangle(
                lower_left=(0.0, 0.0),
                material=Water,  # pyright: ignore
                temperature=20.0,
                size=(1.0, 0.15),
                velocity=(0, 0),
            ),
        ],
    ),
    Configuration(
        dt=1e-4,
        name="Waterjet & Ice Cubes",
        information="Water, Ice -> Water",
        ambient_temperature=10.0,
        geometries=[
            *[
                Circle(
                    is_continuous=True,
                    material=Water,  # pyright: ignore
                    center=(0.04, 0.96),
                    temperature=100.0,
                    velocity=(4, -3),
                    radius=0.025,
                    frame_threshold=i,
                )
                for i in range(1, 200)
            ],
            Rectangle(  # BL
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.495, 0.0),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # BM
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.585, 0.0),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # BR
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.675, 0.0),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # ML
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.5445, 0.09),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # MR
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.6345, 0.09),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
            Rectangle(  # TM
                material=Ice,  # pyright: ignore
                velocity=(0, 0),
                lower_left=(0.585, 0.18),
                size=(0.08, 0.08),
                temperature=-10.0,
            ),
        ],
    ),
    Configuration(
        dt=1e-4,
        name="Waterjet & Snowballs",
        information="Water, Snow -> Water",
        geometries=[
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
            *[
                Rectangle(
                    material=Water,  # pyright: ignore
                    is_continuous=True,
                    frame_threshold=i,
                    lower_left=(0.47, 0.94),
                    velocity=(0, -2),
                    size=(0.06, 0.06),
                )
                for i in range(3, 203)
            ],
            *[
                Circle(
                    material=Water,  # pyright: ignore
                    is_continuous=True,
                    frame_threshold=i,
                    center=(0.5, 0.94),
                    velocity=(0, -2),
                    radius=0.03,
                )
                for i in range(3, 203)
            ],
        ],
    ),
    Configuration(
        dt=1e-4,
        name="Spherefall, Water vs. Ice",
        information="Water, Ice",
        ambient_temperature=0.0,
        geometries=[
            Circle(
                material=Ice,  # pyright: ignore
                center=(0.25, 0.4),
                velocity=(0, -3),
                temperature=-100.0,
                radius=0.1,
            ),
            Circle(
                material=Water,  # pyright: ignore
                center=(0.75, 0.4),
                velocity=(0, -3),
                temperature=-100.0,
                radius=0.1,
            ),
        ],
    ),
]
