from train import GTZAN, CNN, seed_everything
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate import load_checkpoint_and_dispatch


def test(root, batch_size, checkpoint):

    test_dataset = GTZAN(
        root = root,
        split = "test",
        sr = 24000,
    )

    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        num_workers = 8,
        pin_memory=True,
        shuffle=False,
    )

    accelerator = Accelerator()
    device = accelerator.device

    model = CNN()

    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    # load checkpoint
    model = accelerator.load_model(model, checkpoint)

    # valid
    pred_ids = np.array([])
    target_ids = np.array([])
    for audio, target in test_dataloader:
        audio = audio.to(device)
        target = target.to(device)

        with torch.no_grad():
            model.eval()
            output = model(audio)   
            output = accelerator.gather_for_metrics(output)
            target = accelerator.gather_for_metrics(target)

            pred_id = output.argmax(dim=1).cpu().numpy()
            target_id = target.argmax(dim=1).cpu().numpy()

            pred_ids = np.append(pred_ids, pred_id)
            target_ids = np.append(target_ids, target_id)              

    acc = (pred_ids == target_ids).mean()
    accelerator.print("test acc:", f"{acc.item() * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/datasets/gtzan")
    parser.add_argument("--batch_size", type=int, default=54)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/epoch100.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    test(args.root,args.batch_size, args.checkpoint)