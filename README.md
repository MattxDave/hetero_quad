# hetero_quad
Adaptive multi-robot coordination via RL for heterogeneous quadrupeds (Spot + Ghost V60).

Funded by Bowie State I2I Fellowship FY2026. Advisor: Dr. Darsana Josyula.

## Architecture (high-level)

```text
+-------------------------------+      +-------------------------------+
| SpotCommandCenter HTTP API    |      | Ghost move13.py Flask API     |
| Endpoint: :5000 (/api/...)    |      | Endpoint: :5002               |
+---------------+---------------+      +---------------+---------------+
                |                                      |
                +------------------+  +----------------+
                                   v  v
                     +-------------------------------+
                     | abstraction/robot_agent.py    |
                     | Unified RobotAgent interface  |
                     +---------------+---------------+
                                     |
                   +-----------------+-----------------+
                   |                 |                 |
                   v                 v                 v
               policy/             sim/              eval/
```

## Directory layout

```text
hetero_quad/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── abstraction/
│   ├── __init__.py
│   └── robot_agent.py
├── sim/
│   └── __init__.py
├── policy/
│   └── __init__.py
├── eval/
│   └── __init__.py
├── scripts/
│   ├── smoke_test_spot.py
│   └── smoke_test_ghost.py
└── configs/
    └── .gitkeep
```

## Quick start

1. Clone the repository.
2. Create and activate a Python 3.10+ virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create environment file:
   ```bash
   cp .env.example .env
   ```
5. Run smoke checks (only after your robot APIs are available):
   ```bash
   python scripts/smoke_test_spot.py
   python scripts/smoke_test_ghost.py
   ```

## Dependencies

- Python 3.10+
- SpotCommandCenter running and reachable (default `http://localhost:5000/api`)
- `move13.py` Flask service running and reachable (default `http://192.168.168.100:5002`)

## Status

Early development. Grant period: May 21 - Aug 15, 2026.

## License

MIT
