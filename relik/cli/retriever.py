import sys

import hydra
import typer
from omegaconf import OmegaConf

from relik.cli.utils import resolve_config
from relik.common.log import get_logger, print_relik_text_art
from relik.retriever.trainer import train as retriever_train

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def train():
    print_relik_text_art()
    config_dir, config_name, overrides = resolve_config("retriever")

    @hydra.main(
        config_path=str(config_dir),
        config_name=str(config_name),
        version_base="1.3",
    )
    def _retriever_train(conf):
        retriever_train(conf)

    # clean sys.argv for hydra
    sys.argv = sys.argv[:1]
    # add the overrides to sys.argv
    sys.argv.extend(overrides)

    _retriever_train()
