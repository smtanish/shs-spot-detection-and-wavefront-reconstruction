import sys
import numpy as np

from infer_unet import infer_manager
from infer_unet import OUTPUT_DIR



N_ZERNIKES = 15  # change as needed


def main():
    results = infer_manager(
        ref_input=r"C:\Users\tanis\Desktop\unetcnnoffline\perfectspots\IP",
        ab_input=r"C:\Users\tanis\Desktop\unetcnnoffline\perfectspots\IA",
        save_outputs=True,
        output_root=OUTPUT_DIR,
        n_zernike=N_ZERNIKES
    )

    if not results:
        print("⚠️ No valid image pairs were processed.")
        sys.exit(1)

    print(f"\n✅ Number of frames processed: {len(results)}")

    # ---- Inspect structure ----
    print("Keys per result:", results[0].keys())

    # ---- QUICK CHECK: inspect first frame ----
    first = results[0]

    print("\n📌 First frame summary")
    print("Image name:", first["name"])
    print("Ref centroids shape:", first["ref_centroids"].shape)
    print("Ab centroids shape:", first["ab_centroids"].shape)
    print("Displacements shape:", first["displacements"].shape)
    print("Number of matches:", first["num_matches"])

    print("\nFirst 10 displacement vectors:")
    print(first["displacements"][:10])

    # ---- Stack all displacements across frames ----
    all_displacements = [
        r["displacements"] for r in results
        if r is not None and len(r["displacements"]) > 0
    ]

    if not all_displacements:
        print("⚠️ No displacement vectors found in any frame.")
        sys.exit(1)

    stacked = np.vstack(all_displacements)

    print("\n📊 Global displacement statistics")
    print("Total displacement vectors:", stacked.shape)
    print("Mean displacement:", stacked.mean(axis=0))
    print("Std displacement:", stacked.std(axis=0))


if __name__ == "__main__":
    main()
