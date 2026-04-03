import os
import cv2
import gradio as gr
import numpy as np
import torch
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Models configuration
MODELS = {
    "v1.2": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth",
        "arch": "clean",
        "channel_multiplier": 2
    },
    "v1.3": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "arch": "clean",
        "channel_multiplier": 2
    },
    "v1.4": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "arch": "clean",
        "channel_multiplier": 2
    },
    "RestoreFormer": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth",
        "arch": "RestoreFormer",
        "channel_multiplier": 2
    }
}

# Cache for models to avoid re-loading
cached_restorers = {}

def get_restorer(version, upscale, bg_upsampler_type):
    key = f"{version}_{upscale}_{bg_upsampler_type}"
    if key in cached_restorers:
        return cached_restorers[key]

    model_info = MODELS[version]
    
    bg_upsampler = None
    if bg_upsampler_type == "realesrgan":
        if torch.cuda.is_available():
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True
            )
    
    restorer = GFPGANer(
        model_path=model_info["url"],
        upscale=upscale,
        arch=model_info["arch"],
        channel_multiplier=model_info["channel_multiplier"],
        bg_upsampler=bg_upsampler
    )
    
    cached_restorers[key] = restorer
    return restorer

def inference(image, version, upscale, bg_upsample, weight):
    if image is None:
        return None, None
    
    # Convert from RGB to BGR
    input_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    bg_upsampler_type = "realesrgan" if bg_upsample else None
    restorer = get_restorer(version, upscale, bg_upsampler_type)
    
    # Process
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=weight
    )
    
    # Convert results back to RGB
    if restored_img is not None:
        res_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    else:
        res_img = None
        
    res_faces = []
    for face in restored_faces:
        res_faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        
    return res_img, res_faces

# Premium Gradio UI
with gr.Blocks(theme=gr.themes.Soft(), title="GFPGAN Photo Enhancer") as demo:
    gr.Markdown("""
    # 📸 GFPGAN: Practical Face Restoration
    Enhance your old, blurry, or low-resolution photos instantly with State-of-the-Art algorithms.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Input Image")
            
            with gr.Accordion("Advanced Settings", open=True):
                version = gr.Radio(choices=list(MODELS.keys()), value="v1.4", label="Model Version")
                upscale = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Upsampling Scale")
                bg_upsample = gr.Checkbox(label="Enhance Background (Real-ESRGAN)", value=True)
                weight = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Restoration Weight")
            
            submit_btn = gr.Button("Restore Photo", variant="primary")
            
        with gr.Column(scale=1):
            output_img = gr.Image(label="Restored Image")
            output_faces = gr.Gallery(label="Restored Faces", columns=4)

    gr.Examples(
        examples=[["inputs/whole_imgs/00.jpg", "v1.4", 2, True, 0.5]],
        inputs=[input_img, version, upscale, bg_upsample, weight]
    )
    
    submit_btn.click(
        fn=inference,
        inputs=[input_img, version, upscale, bg_upsample, weight],
        outputs=[output_img, output_faces]
    )

if __name__ == "__main__":
    demo.launch()
