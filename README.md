# FLUX LoRA DLC 🥳

https://github.com/user-attachments/assets/09318576-006d-4ccd-8efc-0360c7b289e3

> This repository hosts a Gradio web interface designed to explore the capabilities of the **FLUX.1-dev** diffusion model combined with an extensive collection of community-created **LoRAs (Low-Rank Adaptations)**. It provides an easy way to generate images in various styles defined by these LoRAs, supporting both Text-to-Image and Image-to-Image generation.

The "DLC" (Downloadable Content) in the name playfully refers to the massive, ever-growing library of LoRAs integrated into the app, allowing users to instantly "download" and apply different artistic styles.

## Features

*   **FLUX.1-dev Integration:** Leverages the powerful `black-forest-labs/FLUX.1-dev` model.
*   **Massive LoRA Library:** Includes **over 200+ pre-configured LoRAs** sourced from the Hugging Face Hub community.
*   **LoRA Gallery:** Easily browse and select LoRAs with preview images and titles.
*   **Custom LoRA Support:** Load any compatible FLUX LoRA directly from a Hugging Face repository link.
*   **Text-to-Image Generation:** Create images from text prompts using selected LoRAs.
*   **Image-to-Image Generation:** Modify existing images based on text prompts and LoRA styles.
*   **Real-time Preview:** Utilizes the `TAEF1` tiny autoencoder for a fast preview during the generation process (for Text-to-Image).
*   **User-Friendly Interface:** Built with Gradio for simple interaction.
*   **Adjustable Parameters:** Control steps, CFG scale, seed, dimensions, LoRA scale, and image strength (for I2I).
*   **Performance Monitoring:** Includes basic timing for LoRA loading and image generation.

## Setup and Installation

**Prerequisites:**

*   Python 3.8+
*   `pip` and `git`
*   A CUDA-enabled GPU with sufficient VRAM (at least 16GB recommended, more might be needed depending on resolution) and installed NVIDIA drivers.

**Installation Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PRITHIVSAKTHIUR/FLUX-LoRA-DLC.git
    cd FLUX-LoRA-DLC
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(See `requirements.txt` section below if you need to create it)*

4.  **Hugging Face Login (Optional but Recommended):**
    To ensure seamless access to models and LoRAs from the Hugging Face Hub, log in using your token:
    ```bash
    huggingface-cli login
    ```
    Alternatively, you can uncomment and add your token directly in the script (less secure).

5.  **Run the Gradio App:**
    ```bash
    python app.py
    ```
    *(Assuming your main script is named `app.py`. Adjust if necessary.)*

    The interface should now be accessible via a local URL (usually `http://127.0.0.1:7860`).

### `requirements.txt`

If a `requirements.txt` file is not included, create one with the following content (versions might need adjustment based on compatibility):

```txt
torch --index-url https://download.pytorch.org/whl/cu121 # Or adjust cuXXX for your CUDA version
diffusers>=0.29.0 # Check for latest compatible version
transformers>=4.40.0 # Check for latest compatible version
gradio>=4.0.0
huggingface_hub>=0.20.0
Pillow
numpy
accelerate
safetensors
invisible_watermark
omegaconf
```

## Usage Guide

1.  **Launch the app:** Run `python app.py`.
2.  **Select a LoRA:**
    *   Click on a LoRA card in the gallery. The selected LoRA's Hugging Face repo link will appear above the gallery.
    *   **OR** enter a Hugging Face repository link (e.g., `username/my-flux-lora`) into the "Enter Custom LoRA" textbox and press Enter. A card will appear confirming the loaded custom LoRA and its trigger word (if found).
3.  **Enter Prompt:** Type your desired image description in the "Prompt" box.
    *   **Important:** Check the selected LoRA information or the custom LoRA card. If a "trigger word" is specified, *include it* in your prompt for the LoRA style to activate correctly (e.g., `prompt text, trigger_word`).
4.  **Configure Settings (Optional):**
    *   **Image-to-Image:** Upload an `Input image` under "Advanced Settings" and adjust the `Denoise Strength` (lower values preserve more of the original image).
    *   **Text-to-Image:** Adjust `Steps`, `CFG Scale`, `Width`, `Height`, `LoRA Scale`.
    *   **Seed:** Use the `Seed` slider or check `Randomize seed` for unique results each time.
5.  **Generate:** Click the "Generate" button.
6.  **View Result:**
    *   For Text-to-Image, a real-time preview using TAEF1 will update in the result area, followed by the final high-quality image decoded with the full VAE. A progress bar indicates the steps.
    *   For Image-to-Image, the final image will appear after processing.

## Key Components

*   **Base Model:** `black-forest-labs/FLUX.1-dev`
*   **Preview Autoencoder:** `madebyollin/taef1`
*   **High-Quality Autoencoder:** VAE from `black-forest-labs/FLUX.1-dev`
*   **Core Library:** `diffusers` by Hugging Face
*   **Interface:** Gradio

## LoRA Credits

This application heavily relies on the amazing work of the AI art community who train and share these LoRAs. **Huge thanks to all the creators** whose LoRAs are featured in the default list and available on the Hugging Face Hub!

You can find more FLUX LoRAs here:
[https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.1-dev](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.1-dev)

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions for improvements, bug fixes, or want to add more default LoRAs (please ensure they are compatible and appropriately licensed).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. (You'll need to add an Apache 2.0 LICENSE file to your repo).

