"""
Common Utilities Module.

Provides reusable helper functions that are used across multiple workflows,
eliminating duplication and ensuring consistency.
"""

import logging
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..requirements_parser.schemas import Epic, UserStory, TestContext
from ..utils.unified_config import SystemConfig

logger = logging.getLogger(__name__)


def create_query_from_story(
    epic: Epic,
    user_story: UserStory,
    config: SystemConfig,
) -> str:
    """
    Create search query from epic and user story.
    
    Args:
        epic: Parent epic
        user_story: User story to create query from
        config: System configuration for slicing limits
        
    Returns:
        Formatted query string
        
    Example:
        >>> query = create_query_from_story(epic, story, config)
        >>> print(query)
        Epic: My Epic | Story: My Story | Description: ... | Criteria: ...
    """
    parts = [
        f"Epic: {epic.title}",
        f"Story: {user_story.summary}",
        f"Description: {user_story.description}",
    ]
    
    if user_story.acceptance_criteria:
        max_criteria = config.rag.context_slicing["acceptance_criteria_max"]
        criteria_text = " ".join([
            ac.criteria for ac in user_story.acceptance_criteria[:max_criteria]
        ])
        parts.append(f"Criteria: {criteria_text}")
    
    return " | ".join(parts)


def create_rag_query_from_context(
    context: TestContext,
    config: SystemConfig,
) -> str:
    """
    Create RAG query from test context.
    
    Args:
        context: Test context
        config: System configuration
        
    Returns:
        Formatted query string
        
    Example:
        >>> query = create_rag_query_from_context(context, config)
    """
    parts = [f"Requirement: {context.requirement_text}"]
    
    if context.test_type:
        parts.append(f"Type: {context.test_type}")
    
    return " | ".join(parts)


def format_context_from_tests(
    similar_tests: List[Dict[str, Any]],
    config: SystemConfig,
) -> str:
    """
    Format context string from similar test cases.
    
    Args:
        similar_tests: List of similar test case dictionaries
        config: System configuration for slicing limits
        
    Returns:
        Formatted context string
        
    Example:
        >>> context = format_context_from_tests(tests, config)
        >>> print(context)
        Test 1: ...
        Test 2: ...
    """
    context_parts = []
    
    for i, test in enumerate(similar_tests, 1):
        test_parts = [f"Test {i}:"]
        
        if 'description' in test:
            test_parts.append(f"Description: {test['description']}")
        
        if 'type' in test:
            test_parts.append(f"Type: {test['type']}")
        
        if 'preconditions' in test and test['preconditions']:
            max_preconditions = config.rag.context_slicing.get("preconditions_max", 2)
            preconditions_text = "; ".join(test['preconditions'][:max_preconditions])
            test_parts.append(f"Preconditions: {preconditions_text}")
        
        if 'steps' in test and test['steps']:
            max_steps = config.rag.context_slicing["test_steps_max"]
            steps_text = "; ".join([
                s.get('action', '') for s in test['steps'][:max_steps]
            ])
            test_parts.append(f"Steps: {steps_text}")
        
        if 'expected_results' in test and test['expected_results']:
            max_results = config.rag.context_slicing["expected_results_max"]
            results_text = "; ".join([
                r.get('result', '') for r in test['expected_results'][:max_results]
            ])
            test_parts.append(f"Expected: {results_text}")
        
        context_parts.append(" | ".join(test_parts))
    
    return "\n\n".join(context_parts)


def create_test_text_representation(
    test_data: Dict[str, Any],
    config: SystemConfig,
) -> str:
    """
    Create text representation of test case for embedding.
    
    Args:
        test_data: Test case dictionary
        config: System configuration
        
    Returns:
        Text representation of test
        
    Example:
        >>> text = create_test_text_representation(test_dict, config)
    """
    parts = []
    
    if 'description' in test_data:
        parts.append(f"Test: {test_data['description']}")
    
    if 'type' in test_data:
        parts.append(f"Type: {test_data['type']}")
    
    if 'preconditions' in test_data and test_data['preconditions']:
        max_preconditions = config.rag.context_slicing.get("preconditions_max", 2)
        preconditions_text = "; ".join(test_data['preconditions'][:max_preconditions])
        parts.append(f"Preconditions: {preconditions_text}")
    
    if 'test_steps' in test_data and test_data['test_steps']:
        max_steps = config.rag.context_slicing["test_steps_max"]
        steps = "; ".join([
            s.get('action', '') for s in test_data['test_steps'][:max_steps]
        ])
        parts.append(f"Steps: {steps}")
    
    if 'expected_results' in test_data and test_data['expected_results']:
        max_results = config.rag.context_slicing["expected_results_max"]
        results = "; ".join([
            r.get('result', '') for r in test_data['expected_results'][:max_results]
        ])
        parts.append(f"Expected: {results}")
    
    return " | ".join(parts)


def slice_list_with_config(
    items: List[Any],
    config_key: str,
    config: SystemConfig,
    default: int = 5,
) -> List[Any]:
    """
    Slice list using configuration value.
    
    Args:
        items: List to slice
        config_key: Key in config.rag.context_slicing
        config: System configuration
        default: Default max items if config key not found
        
    Returns:
        Sliced list
        
    Example:
        >>> sliced = slice_list_with_config(steps, "test_steps_max", config)
    """
    max_items = config.rag.context_slicing.get(config_key, default)
    return items[:max_items]


def create_timestamped_directory(
    base_path: Path,
    prefix: str,
    config: SystemConfig,
) -> Path:
    """
    Create timestamped directory.
    
    Args:
        base_path: Base directory path
        prefix: Directory name prefix
        config: System configuration
        
    Returns:
        Created directory path
        
    Example:
        >>> output_dir = create_timestamped_directory(
        ...     Path("results"),
        ...     "test_generation",
        ...     config
        ... )
    """
    if config.output.use_timestamps:
        timestamp = datetime.now().strftime(config.output.timestamp_format)
        dir_name = f"{prefix}_{timestamp}"
    else:
        dir_name = prefix
    
    output_dir = base_path / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug(f"Created directory: {output_dir}")
    return output_dir


def format_separator_line(config: SystemConfig) -> str:
    """
    Create separator line based on configuration.
    
    Args:
        config: System configuration
        
    Returns:
        Separator line string
        
    Example:
        >>> sep = format_separator_line(config)
        >>> print(sep)
        ================================================================================
    """
    return config.output.separator_char * config.output.formatting_width


def format_section_header(
    title: str,
    config: SystemConfig,
    include_separator: bool = True,
) -> str:
    """
    Format section header for output.
    
    Args:
        title: Section title
        config: System configuration
        include_separator: Include separator lines
        
    Returns:
        Formatted header string
        
    Example:
        >>> header = format_section_header("Results", config)
        >>> print(header)
        ================================================================================
        Results
        ================================================================================
    """
    lines = []
    
    if include_separator:
        lines.append(format_separator_line(config))
    
    lines.append(title)
    
    if include_separator:
        lines.append(format_separator_line(config))
    
    return "\n".join(lines)


def extract_checkpoint_step(checkpoint_name: str) -> Optional[int]:
    """
    Extract step number from checkpoint name.
    
    Args:
        checkpoint_name: Checkpoint name (e.g., "step_7566", "checkpoint_step_1000")
        
    Returns:
        Step number or None if not found
        
    Example:
        >>> step = extract_checkpoint_step("checkpoints_hrm_v9_optimized_step_7566")
        >>> print(step)
        7566
    """
    import re
    match = re.search(r'step_(\d+)', checkpoint_name)
    if match:
        return int(match.group(1))
    return None


def safe_dict_get(
    data: Dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        data: Dictionary to search
        *keys: Nested keys to traverse
        default: Default value if not found
        
    Returns:
        Value or default
        
    Example:
        >>> value = safe_dict_get(data, "metadata", "timestamp", default="N/A")
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def merge_dictionaries(
    *dicts: Dict[str, Any],
    deep: bool = False,
) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Dictionaries to merge (later dicts override earlier ones)
        deep: Perform deep merge for nested dicts
        
    Returns:
        Merged dictionary
        
    Example:
        >>> merged = merge_dictionaries(dict1, dict2, dict3)
    """
    if not deep:
        result = {}
        for d in dicts:
            result.update(d)
        return result
    
    # Deep merge with copy.deepcopy (copy imported at module level)
    result = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dictionaries(result[key], value, deep=True)
            else:
                result[key] = copy.deepcopy(value)
    
    return result


def truncate_string(
    text: str,
    max_length: int,
    suffix: str = "...",
) -> str:
    """
    Truncate string to maximum length.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
        
    Example:
        >>> truncated = truncate_string("Long text...", 10)
        >>> print(truncated)
        Long te...
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def batch_list(
    items: List[Any],
    batch_size: int,
) -> List[List[Any]]:
    """
    Split list into batches.
    
    Args:
        items: List to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
        
    Example:
        >>> batches = batch_list(items, batch_size=10)
        >>> for batch in batches:
        ...     process(batch)
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def calculate_statistics(
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate statistics from workflow results.
    
    Args:
        results: Results dictionary
        
    Returns:
        Statistics dictionary
        
    Example:
        >>> stats = calculate_statistics(results)
        >>> print(f"Success rate: {stats['success_rate']}%")
    """
    stats = {}
    
    if "test_cases" in results:
        test_cases = results["test_cases"]
        stats["total_tests"] = len(test_cases) if isinstance(test_cases, list) else 0
    
    if "validation" in results:
        validation = results["validation"]
        if "valid_tests" in validation and "total_tests" in validation:
            total = validation["total_tests"]
            valid = validation["valid_tests"]
            stats["success_rate"] = (valid / total * 100) if total > 0 else 0
    
    if "rag_examples" in results:
        rag_examples = results["rag_examples"]
        if isinstance(rag_examples, dict):
            stats["rag_examples_retrieved"] = sum(len(examples) for examples in rag_examples.values())
    
    return stats


__all__ = [
    "create_query_from_story",
    "create_rag_query_from_context",
    "format_context_from_tests",
    "create_test_text_representation",
    "slice_list_with_config",
    "create_timestamped_directory",
    "format_separator_line",
    "format_section_header",
    "extract_checkpoint_step",
    "safe_dict_get",
    "merge_dictionaries",
    "truncate_string",
    "batch_list",
    "calculate_statistics",
]

