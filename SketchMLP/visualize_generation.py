import os
import numpy as np
from Dataset import GenerationDataset
from Utils import off2abs, draw_three
from Hyper_params import hp
from PIL import Image


def main(output_dir="Data/generation/visualization", num_samples=20):
    os.makedirs(output_dir, exist_ok=True)

    dataset = GenerationDataset()
    if len(dataset) == 0:
        print("[visualize_generation] no samples in GenerationDataset, nothing to visualize.")
        return

    print("[visualize_generation] total samples: {}. saving first {} samples.".format(
        len(dataset), min(num_samples, len(dataset))))

    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        src_cat = sample.get("source_category", "unknown")

        img = sample["sketch_img"]

        np_img = img.numpy().transpose(1, 2, 0)
        np_img = (np_img * 0.5 + 0.5) * 255.0
        np_img = np.clip(np_img, 0, 255).astype("uint8")
        pil_img = Image.fromarray(np_img)

        out_name = "sample_{:04d}_{}.png".format(idx, src_cat)
        out_path = os.path.join(output_dir, out_name)
        pil_img.save(out_path)

    print("[visualize_generation] saved to: {}".format(output_dir))


if __name__ == "__main__":
    main()
