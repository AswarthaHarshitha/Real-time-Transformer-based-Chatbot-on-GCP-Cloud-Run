"""Estimate cost and rough latency for Vertex AI GPU options.

This is a heuristic estimator. Replace cost constants with current GCP pricing for more accuracy.
"""
import argparse

GPU_PRICES = {
    "nvidia-tesla-t4": 0.35,  # $/hour (example)
    "nvidia-tesla-a100": 2.5,
}

# naive latency estimates in ms for a mid-sized model
LATENCY_ESTIMATES = {
    "nvidia-tesla-t4": 60,
    "nvidia-tesla-a100": 20,
}


def estimate(gpu_type: str, instances: int, hours_per_day: float):
    gpu_hour = GPU_PRICES.get(gpu_type, 1.0)
    hourly_cost = gpu_hour * instances
    daily = hourly_cost * hours_per_day
    monthly = daily * 30
    latency = LATENCY_ESTIMATES.get(gpu_type, 100)
    return {"hourly_cost": hourly_cost, "daily_cost": daily, "monthly_cost": monthly, "estimated_latency_ms": latency}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="nvidia-tesla-t4")
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--hours", type=float, default=24.0)
    args = parser.parse_args()
    r = estimate(args.gpu, args.instances, args.hours)
    print(f"Estimated hourly cost: ${r['hourly_cost']:.2f}")
    print(f"Estimated daily cost: ${r['daily_cost']:.2f}")
    print(f"Estimated monthly cost: ${r['monthly_cost']:.2f}")
    print(f"Estimated latency (p50, ms): {r['estimated_latency_ms']}")


if __name__ == '__main__':
    main()
