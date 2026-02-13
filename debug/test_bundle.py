import os
import torch
from pathlib import Path
from monai.bundle import ConfigParser

def run_official_bundle(patient_id="15094"):
    bundle_root = Path("models/wholeBody_ct_segmentation")
    config_file = bundle_root / "configs/inference.json"
    
    # Locate Data
    data_root = Path("../../../venturalitica-sdk-samples-extra/scenarios/surgery-dicom-tcia/data/dicom").resolve()
    patient_dir = data_root / patient_id
    files = sorted(list(patient_dir.rglob("*.dcm")))
    input_files = [str(f) for f in files if "SEG" not in str(f).upper() and "seg" not in str(f).lower()]
    
    # We need to create a temporary NIfTI because the bundle's LoadImaged is configured for NIfTI (usually)
    # Actually DICOM might work if we override the datalist.
    
    parser = ConfigParser()
    parser.read_config(config_file)
    
    # Override bundle_root and datalist
    parser.config["bundle_root"] = str(bundle_root)
    parser.config["datalist"] = [{"image": input_files}]
    parser.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parse only what we need
    preprocessing = parser.get_parsed_content("preprocessing")
    network = parser.get_parsed_content("network")
    inferer = parser.get_parsed_content("inferer")
    
    # Load weights
    weights = bundle_root / "models/model.pt"
    network.load_state_dict(torch.load(weights, map_location="cpu"))
    network.eval()
    network.to(parser.config["device"])
    
    # Prepare data
    data = {"image": input_files}
    prepped = preprocessing(data)
    
    print(f"Preprocessed Image Shape: {prepped['image'].shape}")
    print(f"Preprocessed Affine:\n{prepped['image'].affine}")
    
    inputs = prepped["image"].unsqueeze(0).to(parser.config["device"])
    
    with torch.no_grad():
        output = inferer(inputs, network)
    
    print(f"Output Shape: {output.shape}")
    print(f"Max Logit: {output.max().item()}")
    
    # Argmax
    pred = torch.argmax(output, dim=1, keepdim=True)
    
    # Binary Spine
    spine_mask = ((pred >= 18) & (pred <= 41)) | (pred == 92)
    print(f"Spine Voxels count: {spine_mask.sum().item()}")
    
    # Save a slice for debug
    import matplotlib.pyplot as plt
    ct = inputs[0, 0, :, :, inputs.shape[-1]//2].cpu().numpy()
    mask = spine_mask[0, 0, :, :, inputs.shape[-1]//2].cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ct, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="jet")
    plt.savefig("bundle_debug_axial.png")
    print("Saved bundle_debug_axial.png")

if __name__ == "__main__":
    run_official_bundle()
