# Python SDK Reference

This document covers the Bittensor Python SDK classes and methods.

## Installation

```bash
pip install bittensor
pip install bittensor-wallet
```

## Core Classes

### Subtensor

The main chain interface for sync operations.

```python
from bittensor import Subtensor

# Connect to networks
subtensor = Subtensor(network="finney")      # Mainnet
subtensor = Subtensor(network="test")        # Testnet
subtensor = Subtensor(network="local")       # Localnet
subtensor = Subtensor(network="ws://custom:9944")  # Custom
```

#### Query Methods

```python
# Block information
block = subtensor.get_current_block()
block_hash = subtensor.get_block_hash(block)

# Balance
balance = subtensor.get_balance(coldkey_ss58)
print(f"Balance: {balance.tao} TAO")

# Subnet information
total_subnets = subtensor.get_total_subnets()
subnet_exists = subtensor.subnet_exists(netuid=1)
owner = subtensor.get_subnet_owner(netuid=1)

# Hyperparameters
params = subtensor.get_subnet_hyperparameters(netuid=1)
print(f"Tempo: {params.tempo}")
print(f"Max UIDs: {params.max_allowed_uids}")

# Neuron information
is_registered = subtensor.is_hotkey_registered(netuid=1, hotkey_ss58="5...")
uid = subtensor.get_uid_for_hotkey_on_subnet(netuid=1, hotkey_ss58="5...")
stake = subtensor.get_stake_for_hotkey_on_subnet(netuid=1, hotkey_ss58="5...")

# Metagraph
metagraph = subtensor.metagraph(netuid=1)
```

#### Transaction Methods

```python
from bittensor_wallet import Wallet

wallet = Wallet(name="my_wallet", hotkey="my_hotkey")

# Registration
success = subtensor.register(wallet=wallet, netuid=1)  # PoW
success = subtensor.burned_register(wallet=wallet, netuid=1)  # Burn

# Subnet creation
success, netuid = subtensor.register_network(wallet=wallet)

# Weight setting
success = subtensor.set_weights(
    wallet=wallet,
    netuid=1,
    uids=[0, 1, 2],
    weights=[0.5, 0.3, 0.2],
    wait_for_inclusion=True,
    wait_for_finalization=False
)

# Commit-reveal weights
success = subtensor.commit_weights(
    wallet=wallet,
    netuid=1,
    uids=[0, 1, 2],
    weights=[0.5, 0.3, 0.2]
)

# Staking
success = subtensor.add_stake(
    wallet=wallet,
    hotkey_ss58="5...",
    amount=1.0  # TAO
)

success = subtensor.unstake(
    wallet=wallet,
    hotkey_ss58="5...",
    amount=1.0
)

# Transfer
success = subtensor.transfer(
    wallet=wallet,
    dest="5...",
    amount=1.0
)
```

### AsyncSubtensor

Async version of Subtensor for non-blocking operations.

```python
from bittensor import AsyncSubtensor
import asyncio

async def main():
    async_subtensor = AsyncSubtensor(network="finney")
    
    # Async queries
    block = await async_subtensor.get_current_block()
    balance = await async_subtensor.get_balance(coldkey_ss58)
    
    # Async transactions
    success = await async_subtensor.set_weights(
        wallet=wallet,
        netuid=1,
        uids=uids,
        weights=weights
    )

asyncio.run(main())
```

### Wallet

Key management via bittensor-wallet package.

```python
from bittensor_wallet import Wallet

# Create wallet
wallet = Wallet(name="my_wallet", hotkey="my_hotkey")

# Create keys if needed
wallet.create_if_non_existent()

# Or create explicitly
wallet.create_new_coldkey(n_words=12, use_password=True)
wallet.create_new_hotkey(n_words=12)

# Access addresses
print(f"Coldkey: {wallet.coldkey.ss58_address}")
print(f"Hotkey: {wallet.hotkey.ss58_address}")

# Sign messages
message = b"hello world"
signature = wallet.hotkey.sign(message)

# Verify
is_valid = wallet.hotkey.verify(message, signature)

# Key paths
print(f"Coldkey path: {wallet.coldkeyfile.path}")
print(f"Hotkey path: {wallet.hotkeyfile.path}")
```

### Metagraph

Subnet state snapshot.

```python
from bittensor import Metagraph, Subtensor

# Create metagraph
metagraph = Metagraph(netuid=1, network="finney")

# Sync from chain
subtensor = Subtensor(network="finney")
metagraph.sync(subtensor=subtensor)

# Or sync with lite mode (faster, no weights/bonds)
metagraph.sync(subtensor=subtensor, lite=True)

# Core attributes
print(f"Neurons: {metagraph.n}")
print(f"Block: {metagraph.block}")
print(f"NetUID: {metagraph.netuid}")

# Per-neuron arrays (indexed by UID)
stakes = metagraph.S           # Stake amounts
incentives = metagraph.I       # Incentive scores
dividends = metagraph.D        # Dividend scores
emissions = metagraph.E        # Emission amounts
trust = metagraph.T           # Trust scores
consensus = metagraph.C       # Consensus scores
ranks = metagraph.R           # Rank values
active = metagraph.active     # Activity flags
last_update = metagraph.last_update  # Last weight update block
validator_permit = metagraph.validator_permit
axons = metagraph.axons       # AxonInfo objects
hotkeys = metagraph.hotkeys   # Hotkey addresses
coldkeys = metagraph.coldkeys # Coldkey addresses

# Matrices (full mode only)
weights = metagraph.W         # Weight matrix [validator][miner]
bonds = metagraph.B          # Bond matrix [validator][miner]

# Access specific neuron
uid = 0
print(f"UID {uid}: stake={metagraph.S[uid]}, incentive={metagraph.I[uid]}")

# Find UID by hotkey
hotkey = "5..."
if hotkey in metagraph.hotkeys:
    uid = metagraph.hotkeys.index(hotkey)
```

### Axon

Server component for miners.

```python
from bittensor import Axon

# Create axon
axon = Axon(
    wallet=wallet,
    port=8091,
    ip="0.0.0.0"  # External IP (auto-detected if not specified)
)

# Attach handlers
axon.attach(
    forward_fn=my_forward_function,
    blacklist_fn=my_blacklist_function,  # Optional
    priority_fn=my_priority_function,    # Optional
    verify_fn=my_verify_function         # Optional
)

# Serve on chain (publish endpoint)
axon.serve(
    netuid=1,
    subtensor=subtensor
)

# Start serving
axon.start()

# Stop serving
axon.stop()
```

#### Forward Function Signature

```python
def my_forward_function(synapse: MySynapse) -> MySynapse:
    """
    Process incoming request.
    
    Args:
        synapse: Incoming request with populated request fields
        
    Returns:
        Same synapse with response fields populated
    """
    # Process request
    result = process(synapse.query)
    
    # Set response
    synapse.response = result
    
    return synapse
```

#### Blacklist Function Signature

```python
def my_blacklist_function(synapse: MySynapse) -> tuple[bool, str]:
    """
    Decide whether to reject request.
    
    Args:
        synapse: Incoming request
        
    Returns:
        (should_blacklist, reason)
    """
    caller = synapse.dendrite.hotkey
    
    if caller in blocked_list:
        return True, "Blocked caller"
    
    return False, ""
```

#### Priority Function Signature

```python
def my_priority_function(synapse: MySynapse) -> float:
    """
    Assign priority for request ordering.
    
    Args:
        synapse: Incoming request
        
    Returns:
        Priority value (higher = handled first)
    """
    caller = synapse.dendrite.hotkey
    
    # Prioritize by stake
    if caller in metagraph.hotkeys:
        uid = metagraph.hotkeys.index(caller)
        return float(metagraph.S[uid])
    
    return 0.0
```

### Dendrite

Client component for validators.

```python
from bittensor import Dendrite

# Create dendrite
dendrite = Dendrite(wallet=wallet)

# Single request
response = await dendrite.forward(
    axons=[axon_info],
    synapse=MySynapse(query="test"),
    timeout=12.0
)

# Batch request (multiple miners)
responses = await dendrite.forward(
    axons=[axon_1, axon_2, axon_3],
    synapse=MySynapse(query="test"),
    timeout=12.0
)

# Streaming (if synapse supports)
async for chunk in dendrite.stream(
    axons=[axon_info],
    synapse=StreamingSynapse(query="test")
):
    print(chunk)
```

### Synapse

Request/response data container.

```python
from bittensor import Synapse
from typing import Optional
from pydantic import Field

class MySynapse(Synapse):
    """Custom synapse for my subnet"""
    
    # Request fields (sent by validator)
    query: str = Field(
        ...,
        description="The query to process",
        max_length=10000
    )
    max_tokens: int = Field(
        default=100,
        ge=1,
        le=4096
    )
    
    # Response fields (set by miner)
    response: Optional[str] = None
    confidence: Optional[float] = None
    
    class Config:
        # Required for complex types
        arbitrary_types_allowed = True
        
    def deserialize(self) -> str:
        """Deserialize response for easy access"""
        return self.response
```

#### Built-in Synapse Attributes

Every synapse has these attributes:
```python
synapse.name            # Synapse class name
synapse.timeout         # Request timeout
synapse.total_size      # Total serialized size
synapse.header_size     # Header size

# Dendrite info (set by sender)
synapse.dendrite.hotkey
synapse.dendrite.ip
synapse.dendrite.port
synapse.dendrite.nonce
synapse.dendrite.signature
synapse.dendrite.process_time  # Response time

# Axon info (set by receiver)
synapse.axon.hotkey
synapse.axon.ip
synapse.axon.port
```

---

## Common Patterns

### Registration Check and Auto-Register

```python
def ensure_registered(subtensor, wallet, netuid):
    """Register if not already registered"""
    
    if subtensor.is_hotkey_registered(
        netuid=netuid,
        hotkey_ss58=wallet.hotkey.ss58_address
    ):
        uid = subtensor.get_uid_for_hotkey_on_subnet(
            netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address
        )
        print(f"Already registered as UID {uid}")
        return uid
    
    print("Registering...")
    success = subtensor.burned_register(
        wallet=wallet,
        netuid=netuid
    )
    
    if success:
        uid = subtensor.get_uid_for_hotkey_on_subnet(
            netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address
        )
        print(f"Registered as UID {uid}")
        return uid
    
    raise Exception("Registration failed")
```

### Weight Setting with Rate Limit

```python
class WeightSetter:
    def __init__(self, subtensor, wallet, netuid):
        self.subtensor = subtensor
        self.wallet = wallet
        self.netuid = netuid
        self.last_set_block = 0
        
    async def set_weights_if_ready(self, uids: list, weights: list) -> bool:
        """Set weights respecting rate limit"""
        
        current_block = self.subtensor.get_current_block()
        
        # Get rate limit
        params = self.subtensor.get_subnet_hyperparameters(self.netuid)
        rate_limit = params.weights_rate_limit
        
        if current_block - self.last_set_block < rate_limit:
            blocks_remaining = rate_limit - (current_block - self.last_set_block)
            print(f"Rate limited. {blocks_remaining} blocks remaining.")
            return False
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        success = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True
        )
        
        if success:
            self.last_set_block = current_block
            
        return success
```

### Metagraph Sync Loop

```python
async def metagraph_sync_loop(metagraph, subtensor, interval: int = 60):
    """Keep metagraph synced"""
    
    while True:
        try:
            old_block = metagraph.block
            metagraph.sync(subtensor=subtensor, lite=True)
            
            if metagraph.block != old_block:
                print(f"Synced to block {metagraph.block}")
                
        except Exception as e:
            print(f"Sync error: {e}")
            
        await asyncio.sleep(interval)
```

### Concurrent Miner Queries

```python
async def query_miners_concurrently(
    dendrite,
    metagraph,
    synapse,
    timeout: float = 12.0,
    max_concurrent: int = 50
) -> list:
    """Query miners with concurrency limit"""
    
    import asyncio
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def query_one(axon):
        async with semaphore:
            try:
                return await dendrite.forward(
                    axons=[axon],
                    synapse=synapse,
                    timeout=timeout
                )
            except Exception as e:
                return None
    
    # Get all serving axons
    axons = [
        metagraph.axons[uid] 
        for uid in range(metagraph.n) 
        if metagraph.axons[uid].is_serving
    ]
    
    # Query concurrently
    tasks = [query_one(axon) for axon in axons]
    results = await asyncio.gather(*tasks)
    
    return results
```

---

## Logging

```python
import bittensor as bt

# Set logging level
bt.logging.set_trace()    # Most verbose
bt.logging.set_debug()
bt.logging.set_info()
bt.logging.set_warning()
bt.logging.set_error()

# Log messages
bt.logging.trace("Trace message")
bt.logging.debug("Debug message")
bt.logging.info("Info message")
bt.logging.warning("Warning message")
bt.logging.error("Error message")
```

---

## Configuration

```python
import bittensor as bt

# Access default config
config = bt.Config()

# Add arguments
parser = argparse.ArgumentParser()
bt.Wallet.add_args(parser)
bt.Subtensor.add_args(parser)
bt.Axon.add_args(parser)

# Parse
config = bt.Config(parser)

# Access values
print(config.wallet.name)
print(config.subtensor.network)
print(config.axon.port)
```
