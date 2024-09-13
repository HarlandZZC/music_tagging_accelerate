from train import GTZAN, CNN, seed_everything
import torch
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader
from accelerate import Accelerator


def test(root, device_id, batch_size, checkpoint):

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

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = CNN()
    model = model.to(device)

    # load checkpoint
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint)

    # valid
    pred_ids = np.array([])
    target_ids = np.array([])
    for audio, target in test_dataloader:
        audio = audio.to(device)
        target = target.to(device)

        with torch.no_grad():
            model.eval()
            output = model(audio) 
            pred_id = output.argmax(dim=1).cpu().numpy()
            target_id = target.argmax(dim=1).cpu().numpy()

            pred_ids = np.append(pred_ids, pred_id)
            target_ids = np.append(target_ids, target_id)              

    acc = (pred_ids == target_ids).mean()
    print("test acc:", f"{acc.item() * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/datasets/gtzan")
    parser.add_argument("--device_ids", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/epoch10.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    test(args.root, args.device_ids,args.batch_size, args.checkpoint)