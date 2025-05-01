# DeepGreen

Alex Leccadito
Laura Conti

## Project Description

todo :)

## Setup

### Requirements

To install the project dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### Sentinel Hub Login

- Sign up for an free account at [planet.com](https://www.planet.com/account/)
    - 30'000 token are included in the free trail
- Once registered, store your Client ID and Secret as environment variables:
    - `SENTINEL_HUB_CLIENT_ID`
    - `SENTINEL_HUB_CLIENT_SECRET`

You'll need this login to send catalogue requests (to check which images are available and when) and to download images
from Sentinel Hub.

### Run Code Formatter

```
python -m black .
```
