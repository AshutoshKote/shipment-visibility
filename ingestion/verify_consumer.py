"""Quick sanity check: consume a few pings straight from Kafka.

Independent of Spark -- use this to confirm the simulator -> Kafka leg
works before (or instead of) debugging the Spark Bronze job.

Run:
    python ingestion/verify_consumer.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from confluent_kafka import Consumer


def main(max_msgs=15):
    c = Consumer({
        "bootstrap.servers": config.KAFKA_BOOTSTRAP,
        "group.id": "verify-consumer",
        "auto.offset.reset": "earliest",
    })
    c.subscribe([config.TOPIC_PINGS])
    print(f"[verify] reading up to {max_msgs} messages from '{config.TOPIC_PINGS}'...\n")

    seen = 0
    try:
        while seen < max_msgs:
            msg = c.poll(5.0)
            if msg is None:
                print("[verify] no message in 5s -- is the simulator running?")
                continue
            if msg.error():
                print(f"[verify] error: {msg.error()}")
                continue
            ping = json.loads(msg.value())
            print(f"  p{msg.partition()} off{msg.offset():>5} | "
                  f"{ping['shipment_id']} | ({ping['lat']:.4f},{ping['lon']:.4f}) | "
                  f"{ping['speed_kmh']} km/h | {ping['event_time']}")
            seen += 1
    finally:
        c.close()
        print(f"\n[verify] done. consumed {seen} messages.")


if __name__ == "__main__":
    main()
