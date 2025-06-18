import asyncio
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import orjson
import websockets

# Binance WebSocket URL for BTCUSDT depth updates (100 ms Level-2 Order Book snapshots)
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms" 
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def _json_dumps(msg: dict) -> bytes:
    """
    orjson dumps, newline separated for NDJSON storage.
    """
    return orjson.dumps(msg) + b"\n"

async def stream_depth():
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = RAW_DIR / f"btcusdt_depth_{ts}.jsonl.gz"
    msg_count = 0

    print(f"📡 Connecting to Binance…")
    async with websockets.connect(BINANCE_WS_URL, ping_interval=20) as ws, gzip.open(
        file_path, "wb"
    )as f:
        # Loop through the raw messages from websocket and write to file
        async for raw in ws:
            f.write(_json_dumps(orjson.loads(raw)))
            msg_count += 1
            # Print progress every 1,000 messages
            if msg_count % 1_000 == 0:
                print(f"{msg_count:,} messages saved → {file_path}")

# Main function to run the script
if __name__ == "__main__":
    try:
        asyncio.run(stream_depth())
    except KeyboardInterrupt:
        print("\nStopped by user.")
