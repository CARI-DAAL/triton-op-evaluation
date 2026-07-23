import uvicorn

from triton_evaluation.config import Settings
from triton_evaluation.broker import Broker
from triton_evaluation.api import create_app

app = create_app(Broker.from_settings(Settings.from_env()))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
