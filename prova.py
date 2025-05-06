import numpy as np
import nibabel as nib
from skimage.filters import threshold_otsu
from skimage import morphology as Morpho
from skimage.measure import label, regionprops
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# --- Cargar datos ---
def load_nii(filepath):
    img = nib.load(filepath)
    data = img.get_fdata()
    return data, img.affine

data1, affine1 = load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz")
data_gt, affine_gt = load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz")

# --- Filtro Gaussiano ---
smoothed = gaussian_filter(data1, sigma=0.5)

# --- Umbral de Otsu (modificado) ---
thresh = threshold_otsu(smoothed) - 30
binary = (smoothed > thresh).astype(np.uint8)

# --- Postprocesado con dilatación ---
def postprocess_mask(binary_volume, kernel_size=3, min_area=10, expand_iterations=2):
    kernel = Morpho.cube(kernel_size)
    kernel_2 = Morpho.cube(2)
    opened = Morpho.binary_opening(binary_volume, kernel)
    eroded = Morpho.binary_erosion(opened, kernel_2)
 

    cleaned = np.zeros_like(eroded)
    for i in range(eroded.shape[2]):
        labeled = label(eroded[:, :, i])
        regions = regionprops(labeled)
        if regions:
            largest = max(regions, key=lambda r: r.area)
            if largest.area >= min_area:
                cleaned[:, :, i] = (labeled == largest.label).astype(np.uint8)
    

    return cleaned


cleaned_mask = postprocess_mask(binary)

# --- Métricas ---
def compute_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    eps = 1e-6
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    jaccard = TP / (TP + FP + FN + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)

    return {
        'Dice': dice,
        'Jaccard': jaccard,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity
    }

metrics = compute_metrics(cleaned_mask, data_gt)
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# --- Visualización ---
def visualize_slices(image, smoothed, binary, cleaned_mask, ground_truth, slice_idx=None):
    if slice_idx is None:
        slice_idx = image.shape[2] // 2  # mitad del volumen

    fig, axs = plt.subplots(2, 3, figsize=(10, 12))

    axs[0, 0].imshow(image[:, :, slice_idx], cmap='gray')
    axs[0, 0].set_title("Original")

    axs[0, 1].imshow(smoothed[:, :, slice_idx], cmap='gray')
    axs[0, 1].set_title("Smoothed")

    axs[0, 2].imshow(binary[:, :, slice_idx], cmap='gray')
    axs[0, 2].set_title("Binary (Otsu - 50)")

    axs[1, 0].imshow(cleaned_mask[:, :, slice_idx], cmap='gray')
    axs[1, 0].set_title("Postprocessed Mask")

    axs[1, 1].imshow(ground_truth[:, :, slice_idx], cmap='gray')
    axs[1, 1].set_title("Ground Truth Mask")

    overlap = np.zeros_like(image[:, :, slice_idx], dtype=np.uint8)
    overlap[ground_truth[:, :, slice_idx] == 1] = 1
    overlap[cleaned_mask[:, :, slice_idx] == 1] = 2

    axs[1, 2].imshow(overlap, cmap='jet')
    axs[1, 2].set_title("Prediction vs Ground Truth")

    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_slices(data1, smoothed, binary, cleaned_mask, data_gt)
