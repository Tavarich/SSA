# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import torch.distributed as dist
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    save_dir,
):
    """
    Benchmark inference speed and save prediction masks.
    """
    num_devices = get_world_size()
    total = len(data_loader)
    
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0:
        print(f"\n[INFO] Start inference on {total} batches across {num_devices} devices...")

    num_warmup = min(5, total - 1)

    # --------------------------------------------------------
    # Timers
    # --------------------------------------------------------
    total_compute_time = 0.0
    measured_frames = 0.0

    start_time = time.perf_counter()

    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        iterator = tqdm(data_loader, 
                        total=total, 
                        dynamic_ncols=True, 
                        desc="Inference") if rank == 0 else data_loader
        
        for idx, inputs in enumerate(iterator):
            
           
            # Warmup reset
            if idx == num_warmup:
                start_time = time.perf_counter()

            
            # ====================================================
            # Pure model inference
            # ====================================================
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_compute = time.perf_counter()

            outputs = model(inputs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            if idx >= num_warmup:
                total_compute_time += time.perf_counter() - start_compute

            if outputs is None:
                continue

            # ====================================================
            # Post-process
            # ====================================================
            
            video_len = inputs[0]["length"]
            frames = inputs[0]["file_names"]
            
            if idx >= num_warmup:
                measured_frames += video_len
                
            all_pred_masks = outputs["pred_masks"][0]
            all_masks_np = all_pred_masks.detach().cpu().numpy().astype(np.float32)
            

            save_path = os.path.join(save_dir, inputs[0]["video_name"], inputs[0]["exp_id"])
            os.makedirs(save_path, exist_ok=True)

            for j in range(video_len):
                mask = Image.fromarray(all_masks_np[j] * 255).convert("L")

                file_name = os.path.splitext(os.path.basename(frames[j]))[0]
                mask.save(os.path.join(save_path, file_name + ".png"))




    # ============================================================
    # Final statistics
    # ============================================================
    total_wall_time = time.perf_counter() - start_time
    effective_iters = max(total - num_warmup, 1)

    if is_dist:
        metrics = torch.tensor([
            total_compute_time, 
            total_wall_time,
            measured_frames,
        ], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
        

        dist.reduce(metrics, dst=0, op=dist.ReduceOp.SUM)
        
        if rank != 0:
            return {}
        
        avg_compute_time = metrics[0].item() / num_devices
        avg_wall_time = metrics[1].item() / num_devices
        measured_frames = metrics[2].item()
    else:
        avg_compute_time = total_compute_time
        avg_wall_time = total_wall_time

   
    def format_time(seconds):
        seconds = max(0.0, seconds)
        return str(datetime.timedelta(seconds=int(seconds)))

    compute_iter_time = avg_compute_time / effective_iters
    wall_iter_time = avg_wall_time / effective_iters

    compute_fps = (
        (measured_frames / num_devices) / avg_compute_time
        if avg_compute_time > 0
        else 0.0
    )

    wall_fps = (
        (measured_frames / num_devices) / avg_wall_time
        if avg_wall_time > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Inference Summary (Per Device, {num_devices} GPU{'s' if num_devices > 1 else ''})")
    print("=" * 60)

    print("Pure Model Inference")
    print(f"  Time : {compute_iter_time:.4f} s/iter")
    print(f"  FPS  : {compute_fps:.2f}")

    print("-" * 60)

    print("End-to-End Inference")
    print(f"  Time : {wall_iter_time:.4f} s/iter")
    print(f"  FPS  : {wall_fps:.2f}")
    print("=" * 60 + "\n")
    
    return {}


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
