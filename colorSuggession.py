import argparse
import colorsys
import numpy as np

def parse_rgb(rgb_str: str) -> np.ndarray:
    """
    Parse an RGB string in the format "R,G,B" into a numpy array.
    """
    try:
        parts = [int(x) for x in rgb_str.split(",")]
        if len(parts) != 3:
            raise ValueError("RGB must have 3 values.")
        return np.array(parts, dtype=np.uint8)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid RGB format: {e}")

def rgb_to_hsv(rgb: np.ndarray) -> tuple:
    """
    Convert an RGB array to HSV.
    The input should be in the range 0-255 and output is in normalized [0, 1].
    """
    r, g, b = rgb / 255.0
    return colorsys.rgb_to_hsv(r, g, b)

def hsv_to_rgb(hsv: tuple) -> np.ndarray:
    """
    Convert an HSV tuple (with normalized values) to an RGB array in 0-255.
    """
    r, g, b = colorsys.hsv_to_rgb(*hsv)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)

def get_complementary(rgb: np.ndarray) -> np.ndarray:
    """
    Compute the complementary color by shifting the hue by 180 degrees.
    """
    h, s, v = rgb_to_hsv(rgb)
    comp_h = (h + 0.5) % 1.0
    return hsv_to_rgb((comp_h, s, v))

def get_analogous(rgb: np.ndarray, angle: int = 30) -> tuple:
    """
    Compute two analogous colors by shifting the hue by ±angle degrees.
    """
    h, s, v = rgb_to_hsv(rgb)
    shift = angle / 360.0
    analogous1 = hsv_to_rgb(((h + shift) % 1.0, s, v))
    analogous2 = hsv_to_rgb(((h - shift) % 1.0, s, v))
    return analogous1, analogous2

def get_triadic(rgb: np.ndarray) -> tuple:
    """
    Compute two triadic colors by shifting the hue by ±120 degrees.
    """
    h, s, v = rgb_to_hsv(rgb)
    shift = 120 / 360.0
    triadic1 = hsv_to_rgb(((h + shift) % 1.0, s, v))
    triadic2 = hsv_to_rgb(((h - shift) % 1.0, s, v))
    return triadic1, triadic2

def main():
    parser = argparse.ArgumentParser(
        description="Given an RGB skin tone, suggest harmonious color(s) based on a chosen color theory rule."
    )
    parser.add_argument(
        "--rgb",
        type=str,
        required=True,
        help="Input skin tone as R,G,B (e.g., 210,170,150)"
    )
    parser.add_argument(
        "--rule",
        type=str,
        choices=["complementary", "analogous", "triadic"],
        default="complementary",
        help="Color harmony rule to use for recommendation."
    )
    args = parser.parse_args()

    # Parse input RGB value
    skin_rgb = parse_rgb(args.rgb)
    print(f"Input Skin Tone (RGB): {skin_rgb.tolist()}")

    # Compute recommendations based on chosen rule
    if args.rule == "complementary":
        recommended = get_complementary(skin_rgb)
        print("Recommended Complementary Color (RGB):", recommended.tolist())
    elif args.rule == "analogous":
        rec1, rec2 = get_analogous(skin_rgb)
        print("Recommended Analogous Colors (RGB):", rec1.tolist(), "and", rec2.tolist())
    elif args.rule == "triadic":
        rec1, rec2 = get_triadic(skin_rgb)
        print("Recommended Triadic Colors (RGB):", rec1.tolist(), "and", rec2.tolist())

if __name__ == "__main__":
    main()
