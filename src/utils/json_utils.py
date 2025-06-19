# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import json
import json_repair

logger = logging.getLogger(__name__)


def repair_json_output(content: str) -> str:
    """
    Repair and normalize JSON output.

    Args:
        content (str): String content that may contain JSON

    Returns:
        str: Repaired JSON string, or original content if not JSON
    """
    content = content.strip()
    if content.startswith(("{", "[")) or "```json" in content or "```ts" in content:
        try:
            # If content is wrapped in ```json code block, extract the JSON part
            if content.startswith("```json"):
                content = content.removeprefix("```json")

            if content.startswith("```ts"):
                content = content.removeprefix("```ts")

            if content.endswith("```"):
                content = content.removesuffix("```")

            # Try to repair and parse JSON
            repaired_content = json_repair.loads(content)
            
            # Special handling for Plan objects
            if isinstance(repaired_content, dict):
                # Check if this looks like a partial Plan response
                if "has_enough_context" in repaired_content or "thought" in repaired_content:
                    # Ensure required Plan fields exist with defaults
                    plan_defaults = {
                        "locale": "en-US",
                        "has_enough_context": False,
                        "thought": repaired_content.get("thought", "Unable to process request"),
                        "title": repaired_content.get("title", "Research Plan"),
                        "steps": repaired_content.get("steps", [])
                    }
                    
                    # Merge with existing content, preserving existing values
                    for key, default_value in plan_defaults.items():
                        if key not in repaired_content:
                            repaired_content[key] = default_value
                            logger.info(f"Added missing field '{key}' with default value")
            
            return json.dumps(repaired_content, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"JSON repair failed: {e}")
            logger.debug(f"Content that failed repair: {content[:200]}...")
    return content


def create_fallback_plan(locale: str = "en-US", thought: str = "Unable to process request") -> str:
    """
    Create a minimal fallback Plan JSON when parsing fails completely.
    
    Args:
        locale: The locale to use
        thought: A thought description
        
    Returns:
        Valid Plan JSON string
    """
    fallback_plan = {
        "locale": locale,
        "has_enough_context": False,
        "thought": thought,
        "title": "Fallback Research Plan",
        "steps": []
    }
    return json.dumps(fallback_plan, ensure_ascii=False)


def diagnose_plan_response(content: str) -> dict:
    """
    Diagnose what went wrong with a Plan response.
    
    Args:
        content: The raw LLM response
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnosis = {
        "content_length": len(content),
        "starts_with_brace": content.strip().startswith("{"),
        "ends_with_brace": content.strip().endswith("}"),
        "contains_json_fence": "```json" in content,
        "contains_required_fields": {},
        "is_empty": not content.strip(),
        "appears_to_be_image": "image" in content.lower() and "url" in content.lower(),
        "appears_to_be_error": "error" in content.lower(),
        "first_100_chars": content[:100] if content else "",
    }
    
    required_fields = ["locale", "has_enough_context", "thought", "title", "steps"]
    for field in required_fields:
        diagnosis["contains_required_fields"][field] = f'"{field}"' in content
    
    return diagnosis
