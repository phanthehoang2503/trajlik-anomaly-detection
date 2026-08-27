import argparse
import multiprocessing as mp
import os

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Resumable InvAD training")
    parser.add_argument("--fname", required=True, help="Training config path")
    parser.add_argument(
        "--task", choices=["train", "train_dist"], default="train"
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--port", type=int, default=29500)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", default=None)
    checkpoint_group.add_argument("--init_weights", default=None)
    return parser.parse_args()


def _run_single(config, device, resume, init_weights):
    from src.train_resumable import main

    runtime_config = dict(config)
    runtime_config["meta"] = dict(config["meta"])
    runtime_config["meta"]["device"] = device
    main(runtime_config, resume=resume, init_weights=init_weights)


def _run_distributed(rank, world_size, config, devices, port, resume, init_weights):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import torch
    import torch.distributed as dist

    from src.train_distributed_resumable import main
    from src.utils import init_distributed

    torch.cuda.set_device(0)
    init_distributed(port=port, rank_and_world_size=(rank, world_size))
    try:
        main(config, resume=resume, init_weights=init_weights)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main():
    args = parse_args()
    with open(args.fname, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.task == "train":
        if len(args.devices) != 1:
            raise ValueError("--task train accepts exactly one device")
        _run_single(config, args.devices[0], args.resume, args.init_weights)
        return

    world_size = len(args.devices)
    mp.set_start_method("spawn", force=True)
    processes = []
    for rank in range(world_size):
        process = mp.Process(
            target=_run_distributed,
            args=(
                rank,
                world_size,
                config,
                args.devices,
                args.port,
                args.resume,
                args.init_weights,
            ),
        )
        process.start()
        processes.append(process)

    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(process.exitcode)
    if failed:
        raise RuntimeError(f"Distributed training processes failed: {failed}")


if __name__ == "__main__":
    main()
