# Ancestor local workbench

On Zorin/Ubuntu install the desktop dependency `sudo apt install python3-tk`, then install the project (`python -m pip install -e '.[web]'`, and install Chromium using `python -m playwright install chromium` for web acquisition).

Run:

```sh
my-own-ai workbench --checkpoint /path/to/english.pt --tokenizer /path/to/tokenizer.json --workspace "$PWD" --memory data/module-memory
```

Select Conversation, Author, Engineer, Researcher, Stock Trader or Business Manager. Startup activates none. Selecting a module uses the existing save/unload/load/restore controller transaction. The same Core stays loaded. An unfinished project resumes via Advance one stage, or inspect it via Resume / inspect. Commands run on one serialized worker and selection is disabled during an operation; no module is started automatically. Closing checkpoints and unloads the active module. No training is performed. This is the development workbench, not a qualified product.

The JSON-lines session also registers the six standard modules and supports {"command":"modes"}, alongside switch, start, step and progress. Trusted installed extensions can be explicitly registered with --module PACKAGE:FACTORY; the label falls back to the manifest ID. No discovery of arbitrary installed packages, concurrent module execution, automatic switching, broker permissions or changes to education.

Finance currently provides offline causal replay and library research. Engineering currently provides Python source generation and static checks. Further instrument development and actual neural qualification remain separate. Tk display operation requires desktop testing; the headless tests exercise the same controller and workbench command adapter.
