import numpy as np
import nibabel as nib
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian
from skimage import morphology as morpho
from skimage.measure import label, regionprops
from skimage.segmentation import active_contour
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def load_nii(filepath):
    """Cargar archivo NIfTI y devolver datos y affine matrix"""
    img = nib.load(filepath)
    return img.get_fdata(), img.affine

def adaptive_preprocessing(volume):
    """Preprocesamiento adaptativo por slice"""
    processed = np.zeros_like(volume)
    for z in range(volume.shape[2]):
        slice_img = volume[:,:,z]
        
        # Normalización adaptativa
        p2, p98 = np.percentile(slice_img, (2, 98))
        slice_img = np.clip((slice_img - p2) / (p98 - p2), 0, 1)
        
        # Filtrado adaptativo basado en contenido
        if np.std(slice_img) > 0.1:  # Solo si hay suficiente variación
            slice_img = gaussian(slice_img, sigma=0.5 + 0.5 * np.std(slice_img))
        
        processed[:,:,z] = slice_img
    return processed

def enhanced_thresholding(smoothed_volume):
    """Umbralización mejorada con selección automática de método"""
    binary = np.zeros_like(smoothed_volume)
    for z in range(smoothed_volume.shape[2]):
        slice_img = smoothed_volume[:,:,z]
        
        # Selección automática del método de umbralización
        if np.mean(slice_img) > 0.2 and np.std(slice_img) > 0.1:
            try:
                thresholds = threshold_multiotsu(slice_img, classes=3)
                binary[:,:,z] = np.logical_and(slice_img > thresholds[0], 
                                             slice_img < thresholds[1])
            except:
                thresh = threshold_otsu(slice_img)
                binary[:,:,z] = slice_img > (thresh * 0.9)  # 10% más conservador
        else:
            binary[:,:,z] = slice_img > (np.mean(slice_img) + np.std(slice_img))
    
    return binary

def smart_postprocessing(binary_volume, min_area=50):
    """Postprocesamiento inteligente con contornos activos"""
    processed = np.zeros_like(binary_volume)
    
    for z in range(binary_volume.shape[2]):
        slice_binary = binary_volume[:,:,z]
        
        # Limpieza morfológica adaptativa
        if np.sum(slice_binary) > 100:  # Solo si hay suficiente señal
            # Apertura/clausura con kernel adaptativo
            kernel_size = max(1, int(0.5 + 0.02 * np.sqrt(np.sum(slice_binary))))
            kernel = disk(kernel_size)
            slice_binary = binary_opening(slice_binary, kernel)
            slice_binary = binary_closing(slice_binary, kernel)
        
        # Rellenar huecos y eliminar objetos pequeños
        slice_binary = binary_fill_holes(slice_binary)
        slice_binary = remove_small_objects(slice_binary, min_size=min_area)
        
        # Refinamiento con contornos activos para slices con estructuras
        labeled = label(slice_binary)
        regions = regionprops(labeled)
        if regions:
            largest = max(regions, key=lambda r: r.area)
            if largest.area >= min_area:
                # Crear máscara inicial
                mask = (labeled == largest.label).astype(np.uint8)
                
                # Refinar con contornos activos si la forma es compleja
                if largest.eccentricity > 0.7 or largest.solidity < 0.85:
                    
                        init = np.array(np.where(mask)).T
                        if len(init) > 10:  # Suficientes puntos para el contorno
                            snake = active_contour(gaussian(mask, 3), 
                                                 init, alpha=0.1, beta=0.5,
                                                 gamma=0.01, max_iterations=200)
                            rr, cc = np.round(snake[:, 0]).astype(int), np.round(snake[:, 1]).astype(int)
                            valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
                            refined_mask = np.zeros_like(mask)
                            refined_mask[rr[valid], cc[valid]] = 1
                            refined_mask = binary_fill_holes(refined_mask)
                            slice_binary = refined_mask
                
                        processed[:,:,z] = slice_binary
    
    return processed

def compute_metrics(pred, gt):
    """Cálculo de métricas mejorado"""
    pred = pred.flatten()
    gt = gt.flatten()
    
    # Métricas tradicionales
    metrics = {
        'Dice': f1_score(gt, pred, zero_division=0),
        'Jaccard': jaccard_score(gt, pred),
        'Accuracy': (np.sum(pred == gt) / len(gt)),
        'Precision': precision_score(gt, pred, zero_division=0),
        'Recall': recall_score(gt, pred, zero_division=0),
        'Specificity': np.sum((pred == 0) & (gt == 0)) / (np.sum(gt == 0) + 1e-6)
    }
    
    # Métricas adicionales para evaluación clínica
    if np.any(gt):
        metrics['FP_rate'] = np.sum((pred == 1) & (gt == 0)) / np.sum(gt == 0)
        metrics['FN_rate'] = np.sum((pred == 0) & (gt == 1)) / np.sum(gt == 1)
    
    return metrics

def enhanced_visualization(original, processed, binary, final_mask, ground_truth, slice_idx=None):
    """Visualización mejorada con superposición detallada"""
    if slice_idx is None:
        slice_idx = original.shape[2] // 2

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Primera fila: Proceso de segmentación
    axs[0, 0].imshow(original[:, :, slice_idx], cmap='gray')
    axs[0, 0].set_title("Original Image")
    
    axs[0, 1].imshow(processed[:, :, slice_idx], cmap='gray')
    axs[0, 1].set_title("Preprocessed")
    
    axs[0, 2].imshow(binary[:, :, slice_idx], cmap='gray')
    axs[0, 2].set_title("Initial Segmentation")
    
    # Segunda fila: Resultados y comparación
    axs[1, 0].imshow(final_mask[:, :, slice_idx], cmap='gray')
    axs[1, 0].set_title("Final Segmentation")
    
    axs[1, 1].imshow(ground_truth[:, :, slice_idx], cmap='gray')
    axs[1, 1].set_title("Ground Truth")
    
    # Superposición detallada
    overlay = np.zeros((*original.shape[:2], 3))
    overlay[ground_truth[:, :, slice_idx] == 1] = [0, 1, 0]  # Verde para GT
    overlay[final_mask[:, :, slice_idx] == 1] = [1, 0, 0]    # Rojo para predicción
    overlap = np.logical_and(ground_truth[:, :, slice_idx], final_mask[:, :, slice_idx])
    overlay[overlap] = [1, 1, 0]  # Amarillo para intersección
    
    axs[1, 2].imshow(original[:, :, slice_idx], cmap='gray', alpha=0.7)
    axs[1, 2].imshow(overlay, alpha=0.5)
    axs[1, 2].set_title("Overlap: Green=GT, Red=Pred, Yellow=Match")
    
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- Pipeline principal ---
# Cargar datos
data1, affine1 = load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\image\LIDC-IDRI-0001_R_1.nii.gz")
data_gt, affine_gt = load_nii(r"C:\Users\lclai\Desktop\VOIs\VOIs\nodule_mask\LIDC-IDRI-0001_R_1.nii.gz")

# Preprocesamiento adaptativo
processed_volume = adaptive_preprocessing(data1)

# Segmentación mejorada
binary_volume = enhanced_thresholding(processed_volume)

# Postprocesamiento inteligente
final_mask = smart_postprocessing(binary_volume, min_area=30)

# Cálculo de métricas
metrics = compute_metrics(final_mask, data_gt)
print("\nMétricas de evaluación:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Visualización mejorada
enhanced_visualization(data1, processed_volume, binary_volume, final_mask, data_gt)