import os
import sys
import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy.io as sio
from datetime import datetime

sys.path.append('model')
from model.DP_HAFNet import DP_HAFNet
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 导入数据集类
from utils.dataset import CustomDataset


def parse_args():
    """Parse command line arguments for testing"""
    parser = argparse.ArgumentParser(description="DP_HAFNet Testing Configuration")

    parser.add_argument('--model_name', type=str, default='DP_HAFNet', help='Model name')
    parser.add_argument('--model_path', type=str, default='logs/...', help='Path to trained model weights')

    parser.add_argument('--data_paths', nargs='+',
                        default=[
                            "data/DLdata_breast_24.mat",
                            "data/DLdata_breast_26.mat",
                            "data/DLdata_breast_28.mat",
                            "data/DLdata_long.mat",
                            "data/DLdata_cross.mat",
                            "data/DLdata_EC.mat",
                            "data/DLdata_EP.mat",
                            "data/DLdata_SC.mat",
                            "data/DLdata_SP.mat"
                        ],
                        help='Data file path list')

    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--log_dir', type=str, default='logs', help='Test log save directory')
    parser.add_argument('--save_images', action='store_true', default=True, help='Save prediction images')
    parser.add_argument('--save_mat', action='store_true', default=True, help='Save prediction mat files')

    return parser.parse_args()


def setup_test_logging(args):
    """Set up test logging directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"test_{timestamp}")

    output1_dir = os.path.join(log_dir, "output1")
    output2_dir = os.path.join(log_dir, "output2")
    combined_dir = os.path.join(log_dir, "combined")
    mat_dir = os.path.join(log_dir, "mat_files")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output1_dir, exist_ok=True)
    os.makedirs(output2_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)

    config_file = os.path.join(log_dir, "test_config.txt")
    with open(config_file, 'w') as f:
        f.write("Test Configuration Parameters:\n")
        f.write("=" * 50 + "\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    print(f"Test log directory: {log_dir}")
    print(f"Output1 images directory: {output1_dir}")
    print(f"Output2 images directory: {output2_dir}")
    print(f"Combined images directory: {combined_dir}")
    print(f"Mat files directory: {mat_dir}")

    return log_dir, output1_dir, output2_dir, combined_dir, mat_dir


def save_visualization(data, path):
    """Save visualization of prediction"""
    img_log = 20 * np.log10(np.abs(data) + 0.001)
    img_log = img_log - np.amax(img_log)
    plt.imsave(path, img_log, vmin=-60, vmax=0, cmap="gray", origin="upper")
    plt.close()


def main():
    """Main testing function"""
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir, output1_dir, output2_dir, combined_dir, mat_dir = setup_test_logging(args)
    result_file = os.path.join(log_dir, "test_results.txt")

    net = DP_HAFNet(1, [1, 1]).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    all_metrics = {
        'psnr1': [], 'ssim1': [],
        'psnr2': [], 'ssim2': [],
        'psnr_combined': [], 'ssim_combined': []
    }

    dataset_metrics = {}

    for file_path in args.data_paths:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing dataset: {dataset_name}")

        dataset = CustomDataset([file_path])

        test_data, test_target1, test_target2 = dataset.get_test_data()

        if dataset_name in ["DLdata_breast_24", "DLdata_breast_26", "DLdata_breast_28"]:
            target_combined_index = 75
        else:
            target_combined_index = 77

        import mat73
        full_data = mat73.loadmat(file_path)
        target_combined = full_data['DL_data'][:, :, target_combined_index].astype(np.float32)

        dataset_psnr1, dataset_ssim1 = [], []
        dataset_psnr2, dataset_ssim2 = [], []
        dataset_psnr_combined, dataset_ssim_combined = [], []

        for i in tqdm(range(len(test_data)), desc=f"Testing in {dataset_name}"):
            target1 = test_target1[i].astype(np.float32)
            target2 = test_target2[i].astype(np.float32)

            input_tensor = torch.from_numpy(test_data[i]).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                pred1, pred2 = net(input_tensor.float())

            pred1_np = pred1.squeeze().cpu().numpy()
            pred2_np = pred2.squeeze().cpu().numpy()

            combined_np = pred1_np * pred2_np

            if args.save_mat:
                dataset_mat_dir = os.path.join(mat_dir, dataset_name)
                os.makedirs(dataset_mat_dir, exist_ok=True)
                sio.savemat(os.path.join(dataset_mat_dir, f'test_{i}_output1.mat'), {'DL_data': pred1_np})
                sio.savemat(os.path.join(dataset_mat_dir, f'test_{i}_output2.mat'), {'DL_data': pred2_np})
                sio.savemat(os.path.join(dataset_mat_dir, f'test_{i}_combined.mat'), {'DL_data': combined_np})

            if args.save_images:
                dataset_output1_dir = os.path.join(output1_dir, dataset_name)
                dataset_output2_dir = os.path.join(output2_dir, dataset_name)
                dataset_combined_dir = os.path.join(combined_dir, dataset_name)
                os.makedirs(dataset_output1_dir, exist_ok=True)
                os.makedirs(dataset_output2_dir, exist_ok=True)
                os.makedirs(dataset_combined_dir, exist_ok=True)

                save_visualization(pred1_np, os.path.join(dataset_output1_dir, f'test_{i}.png'))
                save_visualization(pred2_np, os.path.join(dataset_output2_dir, f'test_{i}.png'))
                save_visualization(combined_np, os.path.join(dataset_combined_dir, f'test_{i}.png'))

            psnr1_val = psnr(target1, pred1_np, data_range=pred1_np.max() - pred1_np.min())
            ssim1_val = ssim(target1, pred1_np, data_range=pred1_np.max() - pred1_np.min())

            psnr2_val = psnr(target2, pred2_np, data_range=pred2_np.max() - pred2_np.min())
            ssim2_val = ssim(target2, pred2_np, data_range=pred2_np.max() - pred2_np.min())

            psnr_combined_val = psnr(target_combined, combined_np, data_range=combined_np.max() - combined_np.min())
            ssim_combined_val = ssim(target_combined, combined_np, data_range=combined_np.max() - combined_np.min())

            dataset_psnr1.append(psnr1_val)
            dataset_ssim1.append(ssim1_val)
            dataset_psnr2.append(psnr2_val)
            dataset_ssim2.append(ssim2_val)
            dataset_psnr_combined.append(psnr_combined_val)
            dataset_ssim_combined.append(ssim_combined_val)

        avg_psnr1 = np.mean(dataset_psnr1)
        avg_ssim1 = np.mean(dataset_ssim1)
        avg_psnr2 = np.mean(dataset_psnr2)
        avg_ssim2 = np.mean(dataset_ssim2)
        avg_psnr_combined = np.mean(dataset_psnr_combined)
        avg_ssim_combined = np.mean(dataset_ssim_combined)

        dataset_metrics[dataset_name] = {
            'psnr1': avg_psnr1, 'ssim1': avg_ssim1,
            'psnr2': avg_psnr2, 'ssim2': avg_ssim2,
            'psnr_combined': avg_psnr_combined, 'ssim_combined': avg_ssim_combined
        }

        all_metrics['psnr1'].append(avg_psnr1)
        all_metrics['ssim1'].append(avg_ssim1)
        all_metrics['psnr2'].append(avg_psnr2)
        all_metrics['ssim2'].append(avg_ssim2)
        all_metrics['psnr_combined'].append(avg_psnr_combined)
        all_metrics['ssim_combined'].append(avg_ssim_combined)

        print(f'[{dataset_name}] Output1 - PSNR: {avg_psnr1:.2f}, SSIM: {avg_ssim1:.4f}')
        print(f'[{dataset_name}] Output2 - PSNR: {avg_psnr2:.2f}, SSIM: {avg_ssim2:.4f}')
        print(f'[{dataset_name}] Combined - PSNR: {avg_psnr_combined:.2f}, SSIM: {avg_ssim_combined:.4f}')

    final_psnr1 = np.mean(all_metrics['psnr1'])
    final_ssim1 = np.mean(all_metrics['ssim1'])
    final_psnr2 = np.mean(all_metrics['psnr2'])
    final_ssim2 = np.mean(all_metrics['ssim2'])
    final_psnr_combined = np.mean(all_metrics['psnr_combined'])
    final_ssim_combined = np.mean(all_metrics['ssim_combined'])

    with open(result_file, 'w') as f:
        f.write("Test Results\n")
        f.write("-" * 60 + "\n\n")

        for dataset_name, metrics in dataset_metrics.items():
            f.write(f"[{dataset_name}]\n")
            f.write(f"  Output1: PSNR={metrics['psnr1']:.2f}, SSIM={metrics['ssim1']:.4f}\n")
            f.write(f"  Output2: PSNR={metrics['psnr2']:.2f}, SSIM={metrics['ssim2']:.4f}\n")
            f.write(f"  Combined: PSNR={metrics['psnr_combined']:.2f}, SSIM={metrics['ssim_combined']:.4f}\n\n")

        f.write("-" * 60 + "\n")
        f.write("Overall Averages:\n")
        f.write(f"  Output1 - PSNR: {final_psnr1:.2f}, SSIM: {final_ssim1:.4f}\n")
        f.write(f"  Output2 - PSNR: {final_psnr2:.2f}, SSIM: {final_ssim2:.4f}\n")
        f.write(f"  Combined - PSNR: {final_psnr_combined:.2f}, SSIM: {final_ssim_combined:.4f}\n")

    print(f"\n{'-' * 60}")
    print("Testing completed")
    print(f"Test log saved at: {log_dir}")
    print(f"\nOverall Averages:")
    print(f"  Output1 - PSNR: {final_psnr1:.2f}, SSIM: {final_ssim1:.4f}")
    print(f"  Output2 - PSNR: {final_psnr2:.2f}, SSIM: {final_ssim2:.4f}")
    print(f"  Combined - PSNR: {final_psnr_combined:.2f}, SSIM: {final_ssim_combined:.4f}")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    main()
