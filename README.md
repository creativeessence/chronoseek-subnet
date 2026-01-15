# CHi Subnet Template

A Vibe-codable Bittensor subnet template with Docker-based deployment and automatic updates. 
Made by const <3

## For Subnet Creators

Clone this repo and ask a coding agent: "How do I make a subnet with from this repo @docs/"

## For Validators

All you need:

- Docker and Docker Compose installed
- Bittensor coldkeypub + hotkey on this machine

1. Clone this repository:

```bash
git clone https://github.com/USERNAME/REPO
cd repo
```

2. Copy the example environment file and configure it:

```bash
cp env.example .env
```

3. Start the validator:

```bash
docker-compose down && docker-compose up -d && docker-compose logs -f validator
```

4. View logs:

```bash
docker-compose logs -f validator
```

The validator will automatically update as the subnet owner pushes to main. 