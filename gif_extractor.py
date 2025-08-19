import argparse

from PIL import Image


def extract_frames(gif_path, output_dir="frames"):
    import os

    os.makedirs(output_dir, exist_ok=True)

    with Image.open(gif_path) as im:
        frame_index = 0
        try:
            while True:
                # Copy the frame to avoid modifying the original image
                frame = im.copy()
                frame.save(os.path.join(output_dir, f"frame_{frame_index:03d}.png"))
                frame_index += 1
                im.seek(im.tell() + 1)  # Move to next frame
        except EOFError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a GIF file")
    parser.add_argument("gif_path", type=str, help="Path to the GIF file")
    parser.add_argument("--output_dir", type=str, default="frames", help="Directory to save extracted frames")
    args = parser.parse_args()
    extract_frames(args.gif_path, args.output_dir)
