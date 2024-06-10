from plotly.express import colors


def add_opacity(colorscale):
    for color in colorscale:
        rgb = colors.hex_to_rgb(color[1])
        color[1] = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {max(0.1, color[0])})"

    return colorscale
