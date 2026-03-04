"""
Parallel BLIP2 ITM Client with Multiple Servers

This module provides parallel inference by distributing prompts across multiple BLIP2 servers.
"""

import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .blip2itm import BLIP2ITMClient


class ParallelBLIP2ITMClient:
    """
    Parallel BLIP2 ITM client that distributes prompts across multiple servers.

    Example:
        # Create client with 2 servers
        client = ParallelBLIP2ITMClient(ports=[12182, 12192])

        # Batch inference (parallelized)
        prompts = ["prompt1", "prompt2", "prompt3", "prompt4"]
        scores = client.cosine_batch(image, prompts)

        # Result: [score1, score2, score3, score4]
        # Server 1 processes: prompt1, prompt2
        # Server 2 processes: prompt3, prompt4
        # Total time: ~max(2 prompts) instead of sum(4 prompts)
    """

    def __init__(self, ports: List[int] = None):
        """
        Initialize parallel BLIP2 client.

        Args:
            ports: List of BLIP2 server ports (default: [12182, 12192])
        """
        if ports is None:
            ports = [12182, 12192]

        self.clients = [BLIP2ITMClient(port=port) for port in ports]
        self.num_clients = len(self.clients)

        print(f"[ParallelBLIP2] Initialized with {self.num_clients} servers on ports {ports}")

    def cosine(self, image: np.ndarray, txt: str) -> float:
        """
        Single prompt inference (uses first server for compatibility).

        Args:
            image: RGB image array
            txt: Text prompt

        Returns:
            Cosine similarity score
        """
        return self.clients[0].cosine(image, txt)

    def cosine_batch(self, image: np.ndarray, txt_list: List[str]) -> List[float]:
        """
        Batch inference with parallel processing across multiple servers.

        Args:
            image: RGB image array (same for all prompts)
            txt_list: List of text prompts

        Returns:
            List of cosine similarity scores (same order as txt_list)
        """
        num_prompts = len(txt_list)

        # If only 1 prompt, use single server
        if num_prompts == 1:
            return [self.clients[0].cosine(image, txt_list[0])]

        # Distribute prompts evenly across servers
        tasks = self._distribute_prompts(txt_list)

        # Execute in parallel using ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=self.num_clients) as executor:
            # Submit tasks to each server
            future_to_task = {}
            for client_idx, prompt_indices in tasks.items():
                prompts_subset = [txt_list[i] for i in prompt_indices]

                future = executor.submit(
                    self._process_batch,
                    self.clients[client_idx],
                    image,
                    prompts_subset,
                    prompt_indices
                )
                future_to_task[future] = (client_idx, prompt_indices)

            # Collect results as they complete
            for future in as_completed(future_to_task):
                client_idx, prompt_indices = future_to_task[future]
                try:
                    batch_results = future.result()
                    # Store results with original indices
                    for i, score in enumerate(batch_results):
                        original_idx = prompt_indices[i]
                        results[original_idx] = score
                except Exception as e:
                    print(f"[ParallelBLIP2] Error in client {client_idx}: {e}")
                    # Return 0.0 for failed prompts
                    for idx in prompt_indices:
                        results[idx] = 0.0

        # Return results in original order
        return [results[i] for i in range(num_prompts)]

    def _distribute_prompts(self, txt_list: List[str]) -> dict:
        """
        Distribute prompts across servers for load balancing.

        Strategy: Round-robin distribution
        Example: 4 prompts, 2 servers
            Server 0: [0, 2] (prompt 0, 2)
            Server 1: [1, 3] (prompt 1, 3)

        Args:
            txt_list: List of prompts

        Returns:
            Dict mapping client_idx -> list of prompt indices
        """
        tasks = {i: [] for i in range(self.num_clients)}

        for idx, _ in enumerate(txt_list):
            client_idx = idx % self.num_clients
            tasks[client_idx].append(idx)

        return tasks

    def _process_batch(
        self,
        client: BLIP2ITMClient,
        image: np.ndarray,
        prompts: List[str],
        indices: List[int]
    ) -> List[float]:
        """
        Process a batch of prompts on a single server.

        Args:
            client: BLIP2ITMClient instance
            image: RGB image array
            prompts: List of prompts for this server
            indices: Original indices (for logging)

        Returns:
            List of cosine similarity scores
        """
        if len(prompts) == 1:
            # Single prompt
            score = client.cosine(image, prompts[0])
            return [score]
        else:
            # Multiple prompts - use batch API if available
            try:
                scores = client.cosine_batch(image, prompts)
                return scores
            except Exception as e:
                # Fallback to sequential if batch API fails
                print(f"[ParallelBLIP2] Batch API failed, falling back to sequential: {e}")
                scores = []
                for prompt in prompts:
                    scores.append(client.cosine(image, prompt))
                return scores

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        """
        ITM score inference (uses first server for compatibility).

        Args:
            image: RGB image array
            txt: Text prompt

        Returns:
            ITM score
        """
        return self.clients[0].itm_score(image, txt)

    def ig_score_weighted(self, image: np.ndarray, weights: list = None) -> dict:
        """
        计算 IG Score（在 server 端加权相加）。

        使用第一个 server 来计算 IG score，因为 IG 只有 3 个 prompt，
        并行化收益不大。

        Args:
            image: RGB image array
            weights: 三个 prompt 的权重列表，默认 [1/3, 1/3, 1/3] (等权平均)

        Returns:
            dict: {
                "ig_score": float,       # 加权后的 IG 分数
                "corridor_score": float, # corridor prompt 的原始分数
                "doorway_score": float,  # doorway prompt 的原始分数
                "passage_score": float,  # passage prompt 的原始分数
            }
        """
        return self.clients[0].ig_score_weighted(image, weights)


# Backward compatibility: allow drop-in replacement
class BLIP2ITMClient_Parallel(ParallelBLIP2ITMClient):
    """Alias for backward compatibility"""
    pass
