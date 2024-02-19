# BitDelta Demo

We implemented a minimal demo to show how BitDelta works. This demo allows you to talk to 6 Mistral fine-tuned models together with no more than 30GB GPU memory.

## Requirements

Make sure you have installed the `demo` requirements of the BitDelta repository. If not, you can install them by running the following command:

```bash
pip install -e '.[demo]'
```

in the root directory of the BitDelta repository. Then, move to the `demo` directory:

```bash
cd demo
```

## Download the deltas

We uploaded the deltas of the 6 fine-tuned models to Hugging Face model hub. You can download them by running the following command:

```bash
huggingface-cli download --repo-type model --local-dir checkpoints FasterDecoding/BitDelta_Mistral_combo
```

## Run the demo

For backend, you can run the following command:

```bash
python demo_backend.py
```

For frontend, you can run the following command:

```bash
python demo_gradio.py
```

Then, you can open your browser and visit `http://localhost:7860/` to see the demo.

## Run your own models

If you want to run your own models, you can modify the `supported_models.json` file to point to your own models.
