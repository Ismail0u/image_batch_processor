"""
generate_test_images.py
-----------------------
génération d'images 
"""

import os
import random
import numpy as np
import cv2


def generate_test_images(output_dir: str = "./test_images", count: int = 20, size: int = 512):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Génération de {count} images ({size}×{size}) dans '{output_dir}'...")

    shapes = ["circles", "gradient", "checkerboard", "noise", "lines"]

    for i in range(count):
        kind = shapes[i % len(shapes)]
        img  = np.zeros((size, size, 3), dtype=np.uint8)

        if kind == "circles":
            for _ in range(random.randint(5, 20)):
                cx, cy = random.randint(0, size), random.randint(0, size)
                r      = random.randint(20, 120)
                color  = tuple(random.randint(50, 255) for _ in range(3))
                cv2.circle(img, (cx, cy), r, color, -1)

        elif kind == "gradient":
            for y in range(size):
                for x in range(size):
                    img[y, x] = [
                        int(255 * x / size),
                        int(255 * y / size),
                        int(255 * (1 - x / size)),
                    ]

        elif kind == "checkerboard":
            cell = size // 16
            for y in range(size):
                for x in range(size):
                    val = 255 if ((x // cell) + (y // cell)) % 2 == 0 else 50
                    img[y, x] = [val, val, val]

        elif kind == "noise":
            img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

        elif kind == "lines":
            img[:] = 30
            for _ in range(random.randint(10, 40)):
                pt1 = (random.randint(0, size), random.randint(0, size))
                pt2 = (random.randint(0, size), random.randint(0, size))
                color = tuple(random.randint(100, 255) for _ in range(3))
                cv2.line(img, pt1, pt2, color, random.randint(1, 8))

        # Ajoute du texte pour rendre les images plus réalistes
        cv2.putText(
            img, f"Test #{i+1}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        path = os.path.join(output_dir, f"test_{i+1:03d}.jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"✓ {count} images générées.")
    return output_dir


if __name__ == "__main__":
    generate_test_images(count=30, size=1024)  # 30 images 1024x1024 pour stresser le benchmark