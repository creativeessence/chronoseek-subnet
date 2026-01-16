import httpx
import time
import logging
import bittensor as bt
from typing import List, Tuple
from chronoseek.schemas import VideoSearchRequest, VideoSearchResponse
from chronoseek.scoring import score_response
from chronoseek.epistula import generate_header

logger = logging.getLogger(__name__)

async def query_miner(
    client: httpx.AsyncClient, 
    endpoint: str, 
    request: VideoSearchRequest,
    wallet: bt.Wallet
) -> Tuple[VideoSearchResponse, float]:
    """
    Query a single miner with Epistula signing.
    Returns (Response, Latency).
    """
    start_time = time.time()
    try:
        # Ensure endpoint has scheme
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"
            
        # Generate Epistula headers
        headers = generate_header(wallet.hotkey, request.model_dump())
        
        resp = await client.post(
            f"{endpoint}/search", 
            json=request.model_dump(),
            headers=headers,
            timeout=10.0
        )
        resp.raise_for_status()
        latency = time.time() - start_time
        return VideoSearchResponse(**resp.json()), latency
        
    except Exception as e:
        logger.debug(f"Failed to query miner {endpoint}: {e}")
        return VideoSearchResponse(results=[]), 0.0

async def run_step(
    task_gen,
    metagraph: bt.Metagraph,
    wallet: bt.Wallet,
    client: httpx.AsyncClient
) -> List[Tuple[int, float]]:
    """
    Run a single validation step:
    1. Generate task
    2. Query all miners via HTTP + Epistula
    3. Score responses
    
    Returns: List of (uid, score)
    """
    
    # 1. Generate Task
    video_url, query, ground_truth = task_gen.generate_task()
    logger.info(f"Generated task: {query} for {video_url}")
    
    request_model = VideoSearchRequest(video_url=video_url, query=query)
    
    # 2. Identify Miners with Endpoints
    # In a real subnet, we'd use subtensor.get_commitment(netuid, uid)
    # For simulation/template, we'll assume we can resolve endpoints.
    # If using standard axon info: metagraph.axons[uid].ip + port
    
    scores = []
    
    # Example loop over metagraph (commented out until we have real endpoints)
    # for uid in metagraph.uids:
    #     axon = metagraph.axons[uid]
    #     if axon.ip == "0.0.0.0": continue
    #     
    #     endpoint = f"{axon.ip}:{axon.port}"
    #     resp, latency = await query_miner(client, endpoint, request_model, wallet)
    #     
    #     score = score_response(resp.results, ground_truth, latency)
    #     scores.append((uid, score))
    
    return scores
