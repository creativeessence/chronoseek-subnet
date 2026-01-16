"""
SVMR Subnet Miner.
Exposes the miner logic via a FastAPI endpoint with Epistula verification.
"""

import os
import uvicorn
import bittensor as bt
from fastapi import FastAPI, Request, HTTPException, Depends
from chronoseek.schemas import VideoSearchRequest, VideoSearchResponse
from chronoseek.miner.logic import MinerLogic
from chronoseek.epistula import verify_signature

app = FastAPI()
miner_logic = MinerLogic()

@app.post("/search", response_model=VideoSearchResponse)
async def search(
    request: Request, 
    payload: VideoSearchRequest,
    caller_hotkey: str = Depends(verify_signature)
):
    """
    Handle search requests from validators.
    The verify_signature dependency ensures the request is authenticated.
    """
    bt.logging.info(f"Received request from {caller_hotkey}: {payload.query}")
    
    try:
        # TODO: Check if caller_hotkey is a registered validator with stake
        
        results = miner_logic.search(payload.video_url, payload.query)
        return VideoSearchResponse(results=results)
    except Exception as e:
        bt.logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    # 0. Load configuration
    config = bt.Config()
    config.wallet.name = os.getenv("WALLET_NAME", "default")
    config.wallet.hotkey = os.getenv("HOTKEY_NAME", "default")
    config.netuid = int(os.getenv("NETUID", "1"))
    config.subtensor.network = os.getenv("NETWORK", "finney")

    # Set logging level (default INFO)
    log_level = os.getenv("LOG_LEVEL", "INFO")
    if log_level == "DEBUG":
        bt.logging.set_debug(True)
    elif log_level == "TRACE":
        bt.logging.set_trace(True)
    
    bt.logging.info(f"Starting SVMR Miner on network={config.subtensor.network}, netuid={config.netuid}")


    # 1. Setup Bittensor objects
    wallet = bt.Wallet(config=config)
    ## check if wallet exists (exit if not)
    try:
        if wallet.hotkey:
            bt.logging.info(f"Starting miner with hotkey: {wallet.hotkey.ss58_address}")
    except Exception as e:
        bt.logging.error(f"Error checking wallet: {e}")
        return

    subtensor = bt.Subtensor(config=config)
    metagraph = bt.Metagraph(netuid=config.netuid, network=subtensor.network)
    
    # 2. Check Registration
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(f"Miner hotkey {wallet.hotkey.ss58_address} is NOT registered on netuid {config.netuid}")
        return

    bt.logging.info(f"Miner registered with UID: {metagraph.hotkeys.index(wallet.hotkey.ss58_address)}")
    
    port = int(os.getenv("PORT", "8000"))
    bt.logging.info(f"Starting Miner HTTP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
