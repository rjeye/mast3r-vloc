from matplotlib.colors import CSS4_COLORS, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


class ColorSelector:
    def __init__(self):
        self.color_map = CSS4_COLORS  # Includes a dictionary of named colors in RGB

    def get_color(self, name):
        name = name.lower()
        if name in self.color_map:
            hex_color = self.color_map[name]
            rgb_color = self.hex_to_rgb(hex_color)
            return rgb_color
        else:
            raise ValueError(
                f"Color '{name}' is not available. Check the list of CSS4 named colors."
            )

    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert hex to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_complement(color_rgb):
    """Returns the complementary color of the given RGB color."""
    return tuple(255 - component for component in color_rgb)


class ErrorCmap:
    def __init__(self, error_threshold, cmap_name="summer"):
        self.cmap_name = cmap_name
        self.error_scale = 1.0
        self.error_threshold = error_threshold
        self.error_cmap = self.build_error_cmap()

    def build_error_cmap(self):
        error_cmap = plt.cm.get_cmap(self.cmap_name)(np.linspace(0, 1, 256))[
            :, :3
        ]  # green to yellow
        error_cmap[255] = (0.8, 0, 0)  # red for outliers
        return error_cmap

    def get_error_color(self, error):
        norm_err = max(min(error / self.error_scale, 1.0), 0)
        err_color = self.error_cmap[int(norm_err * 255)] * 255
        return err_color


# # Usage
# selector = ColorSelector()

# # Natural commands
# try:
#     royal_blue_rgb = selector.get_color("royalblue")
#     print(f"Royal Blue RGB: {royal_blue_rgb}")

#     green_rgb = selector.get_color("green")
#     print(f"Green RGB: {green_rgb}")

#     orange_rgb = selector.get_color("orange")
#     print(f"Orange RGB: {orange_rgb}")
# except ValueError as e:
#     print(e)
