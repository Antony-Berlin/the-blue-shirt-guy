"""FastAPI application for the Genesis environment.

The server is a pure evaluator — no tools, no MCP endpoints.
Agents call POST /reset and POST /step via plain HTTP.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required") from e

try:
    from ..models import GenEnvObservation, GenEnvAction
    from .gen_env_environment import GenesisEnvironment
except (ImportError, ModuleNotFoundError):
    from models import GenEnvObservation, GenEnvAction
    from server.gen_env_environment import GenesisEnvironment


app = create_app(
    GenesisEnvironment,
    GenEnvAction,
    GenEnvObservation,
    env_name="gen_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    if args.port == 7860 and args.host == "0.0.0.0":
        main()
    else:
        main(host=args.host, port=args.port)
