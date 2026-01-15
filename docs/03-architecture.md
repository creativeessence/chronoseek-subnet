# Architecture

## Chain Layer (Subtensor)

### What is Subtensor?
Subtensor is Bittensor's blockchain, built on the Substrate framework. It provides:
- Decentralized state management
- Consensus mechanisms
- Economic rule enforcement
- Governance capabilities

### Key Components

#### Pallets
Substrate modules providing specific functionality:

**pallet-subtensor** (Core Pallet):
- Neuron registration logic
- Weight submission and validation
- Emission calculations
- Stake management
- Subnet lifecycle

**pallet-admin-utils**:
- Privileged operations
- Emergency controls
- Sudo commands

**pallet-balances**:
- TAO token management
- Transfer logic

### Storage Items (State)
Everything is stored as key-value pairs in the chain:
- Neuron registrations
- Stake amounts
- Weight matrices
- Subnet hyperparameters
- Block metadata

### Extrinsics (Transactions)
Signed calls that modify chain state:
- `register`: Register a neuron on a subnet
- `set_weights`: Submit validator weights
- `add_stake`: Stake TAO on a hotkey
- `register_network`: Create a new subnet
- `sudo_set_hyperparameters`: Modify subnet parameters

### Events
Notifications emitted when state changes:
- `NeuronRegistered`
- `WeightsSet`
- `StakeAdded`
- `SubnetCreated`

### Block Production
- ~12 second block time on mainnet
- Each block can contain multiple extrinsics
- Block hash = unique identifier

## SDK Layer (Python SDK)

### Purpose
The Python SDK abstracts blockchain complexity:
- Wraps substrate RPC calls
- Manages key signing
- Provides high-level objects (Subtensor, Metagraph, Wallet)
- Implements communication primitives (Axon, Dendrite, Synapse)

### Core Objects

#### Subtensor / AsyncSubtensor
Chain interface for queries and transactions:
```python
from bittensor import Subtensor, AsyncSubtensor

# Sync client
subtensor = Subtensor(network="finney")
block = subtensor.get_current_block()

# Async client  
async_subtensor = AsyncSubtensor(network="finney")
block = await async_subtensor.get_current_block()
```

Key methods:
- `get_current_block()`: Current chain height
- `get_balance()`: TAO balance for coldkey
- `metagraph()`: Get subnet metagraph
- `register_network()`: Create subnet
- `register()`: Register neuron
- `set_weights()`: Submit validator weights
- `add_stake()`: Stake TAO

#### Wallet
Key management:
```python
from bittensor_wallet import Wallet

wallet = Wallet(name="my_wallet", hotkey="my_hotkey")
wallet.create_if_non_existent()

coldkey_address = wallet.coldkey.ss58_address
hotkey_address = wallet.hotkey.ss58_address
```

#### Metagraph
Subnet state snapshot:
```python
from bittensor import Metagraph

metagraph = Metagraph(netuid=1, network="finney")
metagraph.sync()

# Access neuron data
stakes = metagraph.S
incentives = metagraph.I
axons = metagraph.axons
```

### Communication Primitives

#### Axon
Server component run by miners:
```python
from bittensor import Axon

axon = Axon(wallet=wallet, port=8091)
axon.attach(forward_fn=my_handler)
axon.start()
```

Features:
- FastAPI-based HTTP server
- Request/response signing
- Blacklist and priority middleware
- On-chain endpoint publication

#### Dendrite
Client component used by validators:
```python
from bittensor import Dendrite

dendrite = Dendrite(wallet=wallet)
response = await dendrite.forward(
    axons=[axon_info],
    synapse=MySynapse(query="test"),
    timeout=12.0
)
```

Features:
- Async HTTP client
- Request signing
- Batch requests
- Streaming support

#### Synapse
Typed request/response container:
```python
from bittensor import Synapse

class MySynapse(Synapse):
    query: str
    response: Optional[str] = None
```

## Communication Protocol

### ⚠️ SDK Communication Primitives are Legacy

**The Axon/Dendrite/Synapse pattern is considered legacy.** While functional and useful for prototyping, most sophisticated production subnets implement custom communication methods.

**Why subnets move away from SDK primitives:**
- More flexibility in protocols and data formats
- Better performance for specific use cases
- Integration with existing infrastructure
- Custom requirements not covered by SDK

**What miners can commit to chain:**
Miners can store arbitrary connection information in their on-chain metadata:
- Database endpoints
- S3 bucket URLs
- Custom API endpoints  
- IP addresses for any protocol
- Any other discovery information

Validators read this committed data to know how to communicate with each miner.

**What SDK primitives still provide:**
- Convenient hotkey-based message signing
- Built-in nonce/replay protection
- Easy signature verification
- Good for prototyping and simple subnets

### Request Flow (Legacy SDK Pattern)
```
Validator                    Miner
    │                          │
    │   1. Create Synapse      │
    │                          │
    │   2. Sign with Hotkey    │
    │                          │
    │   3. HTTP POST ──────────│
    │                          │
    │   4. Verify Signature    │
    │                          │
    │   5. Process Request     │
    │                          │
    │   6. Sign Response       │
    │                          │
    │   ──────────────────────>│
    │   7. Verify Response     │
    │                          │
```

### Signature Model (When Using SDK)
All requests are cryptographically signed:

Request includes:
- `bt_header_axon_hotkey`: Receiving miner's hotkey
- `bt_header_dendrite_hotkey`: Sending validator's hotkey
- `bt_header_dendrite_nonce`: Unique request nonce
- `bt_header_dendrite_signature`: Signature over headers

Response includes:
- `bt_header_axon_signature`: Miner's response signature

### Production Communication Patterns
**Most production subnets use custom communication.** See Document 04 for:
- HTTP APIs with Epistula headers (custom signing)
- External data source verification
- Socket.io connections
- Custom RPC protocols
- Miners committing S3/database endpoints to chain

## Security

### Replay Protection
- Nonces prevent request replay
- Each request has unique timestamp + random component
- Axons track seen nonces

### Spoofing Prevention
- Hotkey signatures verify sender identity
- Body hashes ensure integrity
- Chain registration links hotkeys to coldkeys

### Key Security
| Key | Location | Security Level |
|-----|----------|----------------|
| Coldkey | Cold storage | Maximum |
| Hotkey | Server | Operational |
| Coldkey password | Memory only | Critical |

### Best Practices
- Never commit keys to repositories
- Use environment variables or secure vaults
- Rotate hotkeys if compromised
- Keep coldkeys offline when possible
