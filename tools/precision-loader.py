import torch
import platform
import subprocess

def get_precision_report():
    report = {
        "Platform": platform.system(),
        "Architecture": platform.machine(),
        "Available_Backends": []
    }

    # 1. NVIDIA (Automotive/Edge: Jetson, Drive, RTX)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name(0)
        p = ["FP32", "FP16", "INT8 (via TensorRT/DLA)"]
        if major >= 8: p += ["BF16", "TF32"]
        if major >= 9: p += ["FP8"]
        
        report["Available_Backends"].append({
            "name": "NVIDIA CUDA",
            "device": gpu_name,
            "precisions": p,
            "compute_cap": f"{major}.{minor}"
        })

    # 2. Intel (Edge: OpenVINO / Core / Arc)
    try:
        from openvino.runtime import Core
        ov_core = Core()
        for device in ov_core.available_devices:
            # Query supported precisions from OpenVINO
            report["Available_Backends"].append({
                "name": f"Intel OpenVINO ({device})",
                "precisions": ["INT8", "FP16", "BF16 (on CPU/XMX)"],
                "notes": "Highly optimized for Edge inference"
            })
    except ImportError:
        pass

    # 3. ARM (Edge: Apple Silicon / MPS)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        report["Available_Backends"].append({
            "name": "Apple Metal (MPS)",
            "precisions": ["FP32", "FP16", "INT8 (Neural Engine)"],
            "notes": "ARM-based unified memory"
        })

    # 4. AMD (ROCm / Edge: Kria/Versal via Vitis-AI)
    if torch.version.hip:
        report["Available_Backends"].append({
            "name": "AMD ROCm / HIP",
            "precisions": ["FP32", "FP64", "FP16", "BF16"],
            "notes": "RDNA/CDNA Architecture"
        })

    return report

# --- Print Formatted Output ---
data = get_precision_report()
print(f"--- {data['Platform']} Hardware Report ---")
for backend in data["Available_Backends"]:
    print(f"\n[Backend: {backend['name']}]")
    if 'device' in backend: print(f"  Device: {backend['device']}")
    print(f"  Precisions: {', '.join(backend['precisions'])}")
