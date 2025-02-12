import concurrent.futures
import pathlib
import os
import subprocess
import shutil
import imghdr


def compress_image(input_path: str, output_path: str):
    subprocess.run(
        f"pngcrush -brute {input_path} {output_path}",
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL  # suppress output
    )


def process_single_image(input_path):
    if imghdr.what(input_path) == "png":
        input_file_size = os.path.getsize(input_path)
        output_path = os.path.join(os.path.dirname(
            input_path), f"compressed_{os.path.basename(input_path)}")
        compress_image(input_path, output_path)
        if not os.path.exists(output_path):
            return None
        output_file_size = os.path.getsize(output_path)

        # replace the original file with the compressed file
        shutil.move(output_path, input_path)

        # calculate bytes saved
        bytes_saved = input_file_size - output_file_size
        return (input_path, bytes_saved)
    return None


if __name__ == "__main__":
    print("Compressing images...")

    public_folder = pathlib.Path("../../public")
    image_paths = []

    # Collect all file paths first
    for root, dirs, files in os.walk(public_folder):
        for file in files:
            input_path = os.path.join(root, file)
            image_paths.append(input_path)

    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_single_image, image_paths)

        # Print results
        for result in results:
            if result:
                input_path, bytes_saved = result
                print(
                    f"Saved {bytes_saved} bytes, {bytes_saved / 1024:.2f} KB, {bytes_saved / 1024 / 1024:.2f} MB for {input_path}")
