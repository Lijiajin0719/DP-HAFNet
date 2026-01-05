import sys
import argparse
import torch
import tqdm
import random
from datetime import datetime
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.append('model')

from utils.dataset import CustomDataset
from utils.loss import get_dual_output_loss
from utils.utils import *
from model.DP_HAFNet import DP_HAFNet


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DP_HAFNet Training Configuration")

    parser.add_argument('--model_name', type=str, default='DP_HAFNet', help='Model name')
    parser.add_argument('--num_epochs', type=int, default=2000, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='Weight decay coefficient')
    parser.add_argument('--lr_period', type=int, default=1000, help='Cosine learning rate period')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate')

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
    parser.add_argument('--log_dir', type=str, default='logs', help='Log save directory')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency (epochs)')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path for resuming training')
    parser.add_argument('--vis_samples', type=int, default=3, help='Number of visualization samples')

    return parser.parse_args()


def setup_logging(args):
    """Set up logging directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{args.model_name}_{timestamp}")

    weight_dir = os.path.join(log_dir, "weights")
    image_dir = os.path.join(log_dir, "images")
    log_file = os.path.join(log_dir, "training_log.txt")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    config_file = os.path.join(log_dir, "config.txt")
    with open(config_file, 'w') as f:
        f.write("Training Configuration Parameters:\n")
        f.write("=" * 50 + "\n")
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    print(f"Log directory: {log_dir}")
    print(f"Weight directory: {weight_dir}")
    print(f"Image directory: {image_dir}")

    return log_dir, weight_dir, image_dir, log_file


def evaluate_model(model, test_loaders, device, epoch, image_dir):
    """Evaluate model performance"""
    model.eval()

    total_psnr1, total_ssim1 = 0, 0
    total_psnr2, total_ssim2 = 0, 0
    total_samples = 0

    vis_indices = random.sample(range(len(test_loaders)), min(len(test_loaders), 3))

    with torch.no_grad():
        progress_bar = tqdm.tqdm(test_loaders, desc='Evaluating')
        for idx, (test_input, test_target1, test_target2) in enumerate(progress_bar):
            test_input = test_input.to(device)
            test_target1 = test_target1.to(device)
            test_target2 = test_target2.to(device)

            test_output1, test_output2 = model(test_input.float())

            output_np1 = test_output1.squeeze().cpu().detach().numpy()
            target_np1 = test_target1.squeeze().cpu().numpy()
            psnr_val1 = psnr(target_np1, output_np1, data_range=output_np1.max() - output_np1.min())
            ssim_val1 = ssim(target_np1, output_np1, data_range=output_np1.max() - output_np1.min())

            output_np2 = test_output2.squeeze().cpu().detach().numpy()
            target_np2 = test_target2.squeeze().cpu().numpy()
            psnr_val2 = psnr(target_np2, output_np2, data_range=output_np2.max() - output_np2.min())
            ssim_val2 = ssim(target_np2, output_np2, data_range=output_np2.max() - output_np2.min())

            total_psnr1 += psnr_val1
            total_ssim1 += ssim_val1
            total_psnr2 += psnr_val2
            total_ssim2 += ssim_val2
            total_samples += 1

            if idx in vis_indices:
                images = [
                    save_visualization(test_input.squeeze().cpu().detach().numpy()),
                    save_visualization(test_target1.squeeze().cpu().detach().numpy()),
                    save_visualization(test_output1.squeeze().cpu().detach().numpy()),
                    save_visualization(test_target2.squeeze().cpu().detach().numpy()),
                    save_visualization(test_output2.squeeze().cpu().detach().numpy())
                ]
                titles = ['Input', 'Target1', 'Output1', 'Target2', 'Output2']
                save_concatenated_image(
                    images,
                    f"epoch_{epoch}_sample_{idx}.png",
                    image_dir,
                    titles
                )

    metrics = {
        'psnr1': total_psnr1 / total_samples,
        'ssim1': total_ssim1 / total_samples,
        'psnr2': total_psnr2 / total_samples,
        'ssim2': total_ssim2 / total_samples,
        'psnr_avg': (total_psnr1 + total_psnr2) / (2 * total_samples),
        'ssim_avg': (total_ssim1 + total_ssim2) / (2 * total_samples)
    }

    return metrics


def train_epoch(model, data_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0

    progress_bar = tqdm.tqdm(data_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        inputs = batch['input'].to(device)
        targets1 = batch['target1'].to(device).float()
        targets2 = batch['target2'].to(device).float()

        outputs1, outputs2 = model(inputs.float())

        loss, loss1, loss2 = get_dual_output_loss(
            outputs1, outputs2,
            targets1, targets2
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()

        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Loss1': f'{loss1.item():.4f}',
            'Loss2': f'{loss2.item():.4f}'
        })

    avg_loss = total_loss / len(data_loader)
    avg_loss1 = total_loss1 / len(data_loader)
    avg_loss2 = total_loss2 / len(data_loader)

    return avg_loss, avg_loss1, avg_loss2


def main():
    """Main training function"""
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    log_dir, weight_dir, image_dir, log_file = setup_logging(args)

    model = DP_HAFNet(1, [1, 1])
    model = model.to(device)

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Successfully resumed training: {args.resume}')

    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)

    dataset = CustomDataset(args.data_paths)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    test_data, test_target1, test_target2 = dataset.get_test_data()
    test_loaders = []
    for i in range(len(test_data)):
        input_tensor = torch.from_numpy(test_data[i]).unsqueeze(0).unsqueeze(0).to(device)
        target_tensor1 = torch.from_numpy(test_target1[i]).unsqueeze(0).unsqueeze(0).to(device)
        target_tensor2 = torch.from_numpy(test_target2[i]).unsqueeze(0).unsqueeze(0).to(device)
        test_loaders.append((input_tensor, target_tensor1, target_tensor2))

    best_metrics = {
        'psnr_avg': 0,
        'ssim_avg': 0,
        'psnr1': 0,
        'psnr2': 0,
        'epoch': 0
    }

    with open(log_file, 'w') as f:
        f.write("Training Log\n")
        f.write("=" * 50 + "\n")

    for epoch in range(1, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}")

        avg_loss, avg_loss1, avg_loss2 = train_epoch(
            model, data_loader, optimizer, device, epoch
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        train_info = (f"Epoch {epoch}/{args.num_epochs}: "
                      f"Loss={avg_loss:.4f} (L1={avg_loss1:.4f}, L2={avg_loss2:.4f}), "
                      f"LR={current_lr:.6f}")
        print(train_info)

        with open(log_file, 'a') as f:
            f.write(train_info + "\n")

        if epoch % args.eval_freq == 0:
            print("Evaluating model...")
            metrics = evaluate_model(model, test_loaders, device, epoch, image_dir)

            eval_info = (f"Evaluation results: "
                         f"PSNR1={metrics['psnr1']:.2f}, SSIM1={metrics['ssim1']:.4f}, "
                         f"PSNR2={metrics['psnr2']:.2f}, SSIM2={metrics['ssim2']:.4f}, "
                         f"PSNR_avg={metrics['psnr_avg']:.2f}, SSIM_avg={metrics['ssim_avg']:.4f}")
            print(eval_info)

            with open(log_file, 'a') as f:
                f.write(eval_info + "\n")

            if metrics['psnr_avg'] > best_metrics['psnr_avg']:
                best_metrics.update(metrics)
                best_metrics['epoch'] = epoch

                torch.save(
                    model.state_dict(),
                    os.path.join(weight_dir, f"{args.model_name}_best.pth")
                )

                best_info = (f"New best model: "
                             f"Epoch={epoch}, "
                             f"PSNR_avg={metrics['psnr_avg']:.2f}, "
                             f"SSIM_avg={metrics['ssim_avg']:.4f}")
                print(best_info)

                with open(log_file, 'a') as f:
                    f.write(best_info + "\n")

            if metrics['psnr1'] > best_metrics['psnr1']:
                torch.save(
                    model.state_dict(),
                    os.path.join(weight_dir, f"{args.model_name}_best_76.pth")
                )
                print(f"New best output1 model: PSNR1={metrics['psnr1']:.2f}")

            if metrics['psnr2'] > best_metrics['psnr2']:
                torch.save(
                    model.state_dict(),
                    os.path.join(weight_dir, f"{args.model_name}_best_77.pth")
                )
                print(f"New best output2 model: PSNR2={metrics['psnr2']:.2f}")

        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(weight_dir, f"{args.model_name}_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metrics': best_metrics,
                'loss': avg_loss,
            }, checkpoint_path)

            print(f"Checkpoint saved: {checkpoint_path}")

    print(f"\n{'-' * 60}")
    print("Training completed ")
    print(f"Best model: Epoch={best_metrics['epoch']}")
    print(f"Best metrics: PSNR_avg={best_metrics['psnr_avg']:.2f}, SSIM_avg={best_metrics['ssim_avg']:.4f}")
    print(f"Log saved at: {log_dir}")
    print(f"\n{'-' * 60}")

    torch.save(
        model.state_dict(),
        os.path.join(weight_dir, f"{args.model_name}_final.pth")
    )


if __name__ == '__main__':
    main()