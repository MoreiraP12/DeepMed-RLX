# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
import re
import sys
from io import StringIO
from typing import Annotated, Dict, List, Union
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from .decorators import log_io

# Initialize REPL, logger, and create an images directory
repl = PythonREPL()
logger = logging.getLogger(__name__)

# Create a public images directory if it doesn't exist
images_dir = os.path.join(os.getcwd(), "web/public/images")
os.makedirs(images_dir, exist_ok=True)


@tool
@log_io
def python_repl_tool(
    code: Annotated[
        str, "The python code to execute to do further analysis or calculation."
    ],
) -> Dict[str, Union[str, List[str]]]:
    """Use this to execute python code and do data analysis or calculation. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. You can create visualizations with matplotlib
    or seaborn and they will be displayed to the user. When saving visualizations, save them to 'web/public/images/' 
    for proper display."""
    if not isinstance(code, str):
        error_msg = f"Invalid input: code must be a string, got {type(code)}"
        logger.error(error_msg)
        
        return {
            "result": f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}",
            "images": []
        }

    logger.info("Executing Python code")
    
    #if code uses matplotlib make sure it includes matplotlib.use('Agg') in case it's not already there
    if "matplotlib" in code:
        if "matplotlib.use('Agg')" not in code:
            code = "import matplotlib\nmatplotlib.use('Agg')\n" + code
    
    #if plt.close() in code remove it
    if "plt.close()" in code:
        code = code.replace("plt.close()", "")
    
    if "plt.show()" in code:
        code = code.replace("plt.show()", "")
        
    # Look for patterns that indicate matplotlib image saving
    images = []
    img_save_pattern = r"savefig\s*\(\s*['\"]([^'\"]+)['\"]"
    img_matches = re.findall(img_save_pattern, code)
    for img_path in img_matches:
        logger.info(f"Detected image save in code: {img_path}")
        # If it's a relative path and not in web/public/images directory, consider it might be saved there
        if not os.path.isabs(img_path) and not img_path.startswith('web/public/images/'):
            img_path = os.path.join('web/public/images', os.path.basename(img_path))
        images.append(img_path)

    try:
        result = repl.run(code)
        # Get stdout output
        logger.info(f"Captured result: {result}")
        
        # Log the captured stdout for debugging
        if result:
            logger.info(f"Captured stdout: {result}")
        else:
            logger.warning("No stdout captured from Python code execution")
                
        # Check if the result is an error message by looking for typical error patterns
        if isinstance(result, str) and ("Error" in result or "Exception" in result):
            logger.error(result)
            return {
                "result": f"Error executing code:\n```python\n{code}\n```\nError: {result}"
            }
        logger.info("Code execution successful")
    except BaseException as e:
        error_msg = repr(e)
        logger.error(error_msg)
        return {
            "result": f"Error executing code:\n```python\n{code}\n```\nError: {error_msg}"
        }

    # Check for IMAGE: markers in the output
    if isinstance(result, str):
        image_pattern = r"IMAGE:\s+(\S+)"
        image_matches = re.findall(image_pattern, result)
        
        # Log image matches for debugging
        logger.info(f"Found image markers in stdout: {image_matches}")
        
        for image_path in image_matches:
            # If the path is relative, make it absolute
            if not os.path.isabs(image_path):
                if not image_path.startswith("web/public/images/"):
                    # Copy the image to the web/public/images directory if it's not already there
                    base_name = os.path.basename(image_path)
                    new_path = os.path.join(images_dir, base_name)
                    if os.path.exists(image_path) and image_path != new_path:
                        import shutil
                        shutil.copy2(image_path, new_path)
                        image_path = f"web/public/images/{base_name}"
            
            logger.info(f"Adding image path to result from stdout: {image_path}")
            images.append(image_path)
            
            # Replace the IMAGE: marker in the output with a more user-friendly message
            result = result.replace(f"IMAGE: {image_path}", f"[Image saved to: {image_path}]")
    
    # If we detected image saving but didn't capture the stdout marker,
    # add a message about the images for the user
    image_messages = []
    for img_path in images:
        logger.info(f"Checking if image exists: {img_path}")
        if os.path.exists(img_path) or os.path.exists(os.path.join(os.getcwd(), img_path)):
            image_messages.append(f"[Image saved to: {img_path}]")
    
    # Add images to stdout output if they weren't already there
    if image_messages and result:
        result += "\n" + "\n".join(image_messages)
    elif image_messages:
        result = "\n".join(image_messages)
    
    # Format the result for better readability, using "Stdout:" instead of "Output:" to match frontend expectations
    result_str = ""
    if result and result.strip():
        result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    else:
        if images:
            image_list = "\n".join([f"[Image saved to: {img}]" for img in images])
            result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {image_list}"
        else:
            result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: (No output)"
    
    # Log the final output for debugging
    logger.info(f"Final result string format: {result_str}")
    logger.info(f"Returning images: {images}")
    
    return {
        "result": result_str,
        "images": images
    }
