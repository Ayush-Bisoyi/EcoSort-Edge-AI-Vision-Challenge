"""
Submission Verification Tool — EcoSort Edge-AI Challenge
==========================================================
Usage:
    python verify_submission.py --model model.onnx
    python verify_submission.py --model model.onnx --val_dir data/public/val
"""

import argparse
import os
import sys
import time
import numpy as np

MAX_FILE_SIZE_BYTES = int(1.5 * 1024 * 1024)
MAX_PARAMETERS = 350_000


def count_onnx_parameters(onnx_model_path: str) -> int:
    import onnx
    model = onnx.load(onnx_model_path)
    total_params = 0
    for initializer in model.graph.initializer:
        dims = initializer.dims
        param_count = 1
        for d in dims:
            param_count *= d
        total_params += param_count
    return total_params


def verify_submission(onnx_path: str, val_dir: str = ""):
    print("=" * 60)
    print(" 🛠️  EcoSort Pre-Submission Verification")
    print("=" * 60)

    if not os.path.exists(onnx_path):
        print(f"❌ Error: Model file not found at '{onnx_path}'")
        sys.exit(1)

    all_passed = True

    file_size_bytes = os.path.getsize(onnx_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    size_passed = file_size_bytes <= MAX_FILE_SIZE_BYTES
    all_passed = all_passed and size_passed
    print(f"1. Model File Size:    {file_size_mb:.2f} MB / 1.50 MB limit -> {'PASS [OK]' if size_passed else 'FAIL [EXCEEDED]'}")

    try:
        import onnx
        param_count = count_onnx_parameters(onnx_path)
        params_passed = param_count <= MAX_PARAMETERS
        all_passed = all_passed and params_passed
        print(f"2. Total Parameters:   {param_count:,} / {MAX_PARAMETERS:,} limit -> {'PASS [OK]' if params_passed else 'FAIL [EXCEEDED]'}")
    except ImportError:
        print("2. Total Parameters:   [WARNING] 'onnx' library not installed. Install via: pip install onnx")

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        input_name = inputs[0].name
        input_shape = inputs[0].shape
        output_name = outputs[0].name
        output_shape = outputs[0].shape

        print(f"3. Input Signature:    Name='{input_name}', Shape={input_shape}")
        print(f"4. Output Signature:   Name='{output_name}', Shape={output_shape}")

        channels = input_shape[1] if len(input_shape) == 4 else None
        h = input_shape[2] if len(input_shape) == 4 else None
        w = input_shape[3] if len(input_shape) == 4 else None
        num_classes = output_shape[1] if len(output_shape) == 2 else None

        shape_valid = (channels == 3 and h == 96 and w == 96 and num_classes == 6)
        all_passed = all_passed and shape_valid
        print(f"   Shape Validation:   {'PASS [OK]' if shape_valid else 'FAIL [Expected: (Batch, 3, 96, 96) -> (Batch, 6)]'}")

        dummy = np.random.randn(1, 3, 96, 96).astype(np.float32)
        for _ in range(5):
            _ = session.run([output_name], {input_name: dummy})
        
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = session.run([output_name], {input_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000.0)
        print(f"5. Avg CPU Latency:    {np.mean(latencies):.2f} ms/image (Target: <= 35 ms)")

    except ImportError:
        print("3-5. ONNX Runtime:     [WARNING] 'onnxruntime' not installed. Install via: pip install onnxruntime")

    if val_dir and os.path.exists(val_dir):
        print(f"\nEvaluating on local validation directory: '{val_dir}'...")
        try:
            from PIL import Image
            classes = sorted(os.listdir(val_dir))
            class_to_idx = {c: i for i, c in enumerate(classes) if os.path.isdir(os.path.join(val_dir, c))}
            
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            
            y_true, y_pred = [], []
            for c_name, c_idx in class_to_idx.items():
                folder = os.path.join(val_dir, c_name)
                for f in os.listdir(folder):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        img = Image.open(os.path.join(folder, f)).convert("RGB").resize((96, 96))
                        arr = np.array(img).astype(np.float32) / 255.0
                        arr = np.transpose(arr, (2, 0, 1))
                        arr = (arr - mean) / std
                        arr = np.expand_dims(arr, 0)
                        logits = session.run([output_name], {input_name: arr})[0]
                        y_true.append(c_idx)
                        y_pred.append(int(np.argmax(logits, axis=1)[0]))

            if y_true:
                acc = np.mean(np.array(y_true) == np.array(y_pred)) * 100
                print(f"6. Local Val Accuracy: {acc:.2f}% ({len(y_true)} samples)")
        except Exception as e:
            print(f"Validation evaluation error: {e}")

    print("=" * 60)
    if all_passed:
        print("🎉 ALL CONSTRAINT CHECKS PASSED! Your file is ready for submission.")
    else:
        print("⚠️ SOME CHECKS FAILED. Please fix the violations before submitting.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-submission ONNX verification")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to your model.onnx")
    parser.add_argument("--val_dir", type=str, default="", help="Optional path to validation dataset folder")
    args = parser.parse_args()
    verify_submission(args.model, args.val_dir)
