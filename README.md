# DiffusionCLIP for Traffic Sign Manipulation

This repository provides a complete pipeline for adapting and fine-tuning the **DiffusionCLIP** model for the task of traffic sign image manipulation. The workflow allows you to train the model to apply specific environmental and weather effects (e.g., 'snowy', 'rainy', 'night') and then interactively test your results using a bundled Gradio web demo.

This project integrates the official PyTorch implementation of [DiffusionCLIP (CVPR 2022)](https://www.google.com/search?q=https://github.com/gigatech/rom-diffusion) as a submodule for the core training and inference logic.

## 📋 Core Workflow

The process is designed to be straightforward:

1.  **Setup Environment:** Clone the repository (including the submodule) and install all dependencies.
2.  **Download Prerequisites:** Fetch the base pre-trained diffusion model and the traffic sign dataset.
3.  **Fine-tune DiffusionCLIP:** Run the training script to teach the model a new manipulation, such as making a traffic sign appear "snowy". This will generate a custom `.pth` checkpoint.
4.  **Launch the Gradio Demo:** Copy your newly trained checkpoint to the demo directory and launch the interactive web UI to see your model in action.

## ⚙️ 1. Setup and Installation

### a) Clone the Repository

First, clone this repository and its submodule (`DiffusionCLIP`):

```bash
git clone --recurse-submodules <your-repository-url>
cd <repository-name>
```

### b) Install Dependencies

This project has two sets of requirements: one for the core DiffusionCLIP model and another for the Gradio demo. It's recommended to install both.

```bash
# Install requirements for the main training script
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Install requirements for the Gradio demo app
pip install -r gradio-demo-app/requirements.txt
```

### c) Download Base Model & Dataset

Before you can fine-tune, you need the base pre-trained models and the image data.

1.  **Base Diffusion Model:** The fine-tuning process starts with a powerful 512x512 diffusion model pre-trained on ImageNet.

    ```bash
    # Create the directory for pretrained models
    mkdir -p pretrained

    # Download the model from OpenAI's official source
    wget -O ./pretrained/512x512_diffusion.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt
    ```

2.  **Traffic Sign Dataset:** The training and demo scripts expect the traffic sign dataset to be in the `./data/traffic_Data` directory. Please download the dataset and place it accordingly. You will also need to generate `train.csv` and `val.csv` files. A helper script is provided:

    ```bash
    # Ensure your data is in ./data/traffic_Data/DATA
    # Then run the script to create the CSV splits
    python scripts/generate_train_val_csv.py
    ```

## 🚀 2. Fine-tuning DiffusionCLIP on Traffic Signs

Now you are ready to train the model. The goal is to create a specialized checkpoint for a single text-guided manipulation.

**Example: Fine-tuning for a "snowy" effect**

Run the `main.py` script with the following arguments. This command fine-tunes the base model using the source text "traffic sign" and the target text "traffic sign in snowy weather".

```bash
python main.py \
    --clip_finetune \
    --config traffic_sign.yml \
    --exp ./runs/traffic_sign_snowy \
    --edit_attr traffic_sign_snowy \
    --do_train 1 \
    --do_test 0 \
    --n_train_img 100 \
    --n_iter 10 \
    --t_0 301 \
    --n_inv_step 40 \
    --n_train_step 6 \
    --lr_clip_finetune 8e-6 \
    --l1_loss_w 1.0
```

  - `--config traffic_sign.yml`: Specifies the model and data configuration for our 512x512 traffic sign task.
  - `--edit_attr traffic_sign_snowy`: A key defined in `utils/text_dic.py` that maps to the source/target text pair.
  - `--exp`: The directory where training logs and results will be saved.
  - `--n_iter`: The number of training epochs.

Upon completion, a new fine-tuned model will be saved in the `./checkpoint/` directory. The filename will be long and descriptive, for example: `traffic_sign_snowy_FT_...-9.pth`. This is the file you will use in the demo.

## 🖥️ 3. Running the Gradio Demo

After fine-tuning, you can use the Gradio app to interactively test your model.

### a) Move Your Trained Checkpoint

The Gradio app looks for models in its own `checkpoints` directory. Copy the model you just trained into it.

```bash
# Example:
cp ./checkpoint/traffic_sign_snowy_FT_...-9.pth ./gradio-demo-app/checkpoints/
```

*(**Note:** You must copy a checkpoint for each effect you want to test in the demo, e.g., for 'rainy', 'foggy', etc.)*

### b) Launch the Application

Navigate into the demo directory and run the `app.py` script.

```bash
cd gradio-demo-app
python src/app.py
```

The script will pre-load all available models from the `gradio-demo-app/checkpoints/` directory. Once loaded, open the local URL provided in your terminal (e.g., `http://127.0.0.1:7860`) to access the web interface and test your custom traffic sign manipulations.
