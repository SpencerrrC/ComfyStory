"""
ComfyUI workflow templates for different generation scenarios.
"""

import json
from typing import Dict, Any

class ComfyUIWorkflows:
    """Manager for ComfyUI workflow templates"""
    
    @staticmethod
    def text_to_image_workflow(prompt: str, negative_prompt: str = "low quality, bad quality, sketches", 
                              width: int = 512, height: int = 512, seed: int = -1) -> Dict[str, Any]:
        """
        Basic text-to-image workflow for Stable Diffusion.
        
        Args:
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            width: Image width
            height: Image height
            seed: Random seed (-1 for random)
            
        Returns:
            ComfyUI workflow dictionary
        """
        workflow = {
            "3": {
                "inputs": {
                    "seed": seed if seed != -1 else 156680208700286,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "v1-5-pruned.ckpt"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "5": {
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "ComfyStory",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        return workflow
    
    @staticmethod
    def image_to_image_workflow(prompt: str, image_path: str, denoise_strength: float = 0.75,
                               negative_prompt: str = "low quality, bad quality", seed: int = -1) -> Dict[str, Any]:
        """
        Image-to-image workflow for modifying existing images.
        
        Args:
            prompt: Positive prompt text
            image_path: Path to input image
            denoise_strength: How much to change the image (0-1)
            negative_prompt: Negative prompt text
            seed: Random seed (-1 for random)
            
        Returns:
            ComfyUI workflow dictionary
        """
        workflow = {
            "1": {
                "inputs": {
                    "image": image_path,
                    "upload": "image"
                },
                "class_type": "LoadImage"
            },
            "2": {
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEEncode"
            },
            "3": {
                "inputs": {
                    "seed": seed if seed != -1 else 156680208700286,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": denoise_strength,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["2", 0]
                },
                "class_type": "KSampler"
            },
            "4": {
                "inputs": {
                    "ckpt_name": "v1-5-pruned.ckpt"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "7": {
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "filename_prefix": "ComfyStory_img2img",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        return workflow
    
    @staticmethod
    def upscale_workflow(image_path: str, scale_factor: float = 2.0) -> Dict[str, Any]:
        """
        Workflow for upscaling images using ESRGAN or similar models.
        
        Args:
            image_path: Path to input image
            scale_factor: Upscale factor
            
        Returns:
            ComfyUI workflow dictionary
        """
        workflow = {
            "1": {
                "inputs": {
                    "image": image_path,
                    "upload": "image"
                },
                "class_type": "LoadImage"
            },
            "2": {
                "inputs": {
                    "upscale_model": ["3", 0],
                    "image": ["1", 0]
                },
                "class_type": "ImageUpscaleWithModel"
            },
            "3": {
                "inputs": {
                    "model_name": "RealESRGAN_x4plus.pth"
                },
                "class_type": "UpscaleModelLoader"
            },
            "4": {
                "inputs": {
                    "filename_prefix": "ComfyStory_upscaled",
                    "images": ["2", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        return workflow 