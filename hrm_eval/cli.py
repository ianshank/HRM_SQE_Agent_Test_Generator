"""
Command-line interface for HRM SQE Agent Test Generator.

Provides CLI commands for:
- Running the API server
- Processing requirements
- Managing the drop folder
- Running evaluations

NO HARDCODED VALUES - all configuration through files or environment.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_logging_from_args(verbose: bool = False, quiet: bool = False) -> None:
    """
    Configure logging based on CLI arguments.

    Args:
        verbose: Enable verbose (DEBUG) logging
        quiet: Suppress all but error messages
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the API server."""
    import uvicorn
    from .utils.config import get_settings

    settings = get_settings()

    host = args.host or settings.api.host
    port = args.port or settings.api.port
    workers = args.workers or settings.api.workers
    reload = args.reload or settings.is_development()

    logger.info(f"Starting API server on {host}:{port}")

    uvicorn.run(
        "hrm_eval.api_service.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="debug" if args.verbose else "info",
    )

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate test cases from requirements file."""
    from .utils.config import get_settings
    from .requirements_parser import RequirementParser
    from .test_generator import TestCaseGenerator
    import json

    settings = get_settings()
    input_file = Path(args.input)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    logger.info(f"Processing requirements from: {input_file}")

    # Load requirements
    with open(input_file) as f:
        if input_file.suffix == ".json":
            requirements = json.load(f)
        else:
            # Treat as natural language
            from .requirements_parser.nl_parser import NaturalLanguageParser
            nl_parser = NaturalLanguageParser()
            requirements = nl_parser.parse_to_epic(f.read())

    # Parse and generate
    parser = RequirementParser()
    generator = TestCaseGenerator(
        model_path=str(settings.get_model_path()),
        device=settings.get_device(),
    )

    test_contexts = parser.extract_test_contexts(requirements)
    test_cases = generator.generate_test_cases(test_contexts)

    # Output results
    output_file = args.output or input_file.with_suffix(".test_cases.json")

    results = {
        "source": str(input_file),
        "test_cases": [
            {
                "id": tc.id,
                "description": tc.description,
                "type": tc.type.value if hasattr(tc.type, "value") else str(tc.type),
                "priority": tc.priority.value if hasattr(tc.priority, "value") else str(tc.priority),
            }
            for tc in test_cases
        ],
        "count": len(test_cases),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Generated {len(test_cases)} test cases to: {output_file}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Start the drop folder watcher."""
    from .drop_folder.cli import main as drop_folder_main

    # Pass through to drop folder CLI
    sys.argv = ["hrm-drop-folder", "watch"]
    if args.input_dir:
        sys.argv.extend(["--input-dir", args.input_dir])
    if args.output_dir:
        sys.argv.extend(["--output-dir", args.output_dir])

    return drop_folder_main()


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate model checkpoint."""
    from .evaluation import Evaluator
    from .utils.config import get_settings

    settings = get_settings()
    checkpoint = args.checkpoint or settings.model.default_checkpoint

    logger.info(f"Evaluating checkpoint: {checkpoint}")

    evaluator = Evaluator()
    results = evaluator.evaluate(checkpoint_name=checkpoint)

    print("\nEvaluation Results:")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Accuracy: {results.get('accuracy', 'N/A')}")
    print(f"  Solve Rate: {results.get('solve_rate', 'N/A')}")

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show current configuration."""
    from .utils.config import get_settings, validate_configuration
    import json

    settings = get_settings()
    warnings = validate_configuration(settings)

    if args.json:
        config_dict = settings.model_dump(exclude={"openai_api_key", "anthropic_api_key", "pinecone_api_key", "wandb_api_key", "security"})
        print(json.dumps(config_dict, indent=2, default=str))
    else:
        print("\nCurrent Configuration:")
        print(f"  Environment: {settings.env}")
        print(f"  Debug: {settings.debug}")
        print(f"  Log Level: {settings.log_level}")
        print(f"\nAPI:")
        print(f"  Host: {settings.api.host}")
        print(f"  Port: {settings.api.port}")
        print(f"  Workers: {settings.api.workers}")
        print(f"\nModel:")
        print(f"  Path: {settings.model.model_path}")
        print(f"  Device: {settings.get_device()}")
        print(f"\nRAG:")
        print(f"  Backend: {settings.rag.backend}")
        print(f"  Collection: {settings.rag.collection_name}")
        print(f"\nGeneration:")
        print(f"  Mode: {settings.generation.mode}")
        print(f"  Strategy: {settings.generation.merge_strategy}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    print("HRM SQE Agent Test Generator")
    print("Version: 1.0.0")
    print("Python: " + sys.version.split()[0])

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch: Not installed")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="hrm-eval",
        description="HRM SQE Agent Test Generator - AI-powered test case generation",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run the API server")
    serve_parser.add_argument("--host", help="Host address")
    serve_parser.add_argument("--port", type=int, help="Port number")
    serve_parser.add_argument("--workers", type=int, help="Number of workers")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.set_defaults(func=cmd_serve)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate test cases")
    gen_parser.add_argument("input", help="Input requirements file")
    gen_parser.add_argument("-o", "--output", help="Output file")
    gen_parser.set_defaults(func=cmd_generate)

    # watch command
    watch_parser = subparsers.add_parser("watch", help="Start drop folder watcher")
    watch_parser.add_argument("--input-dir", help="Input directory to watch")
    watch_parser.add_argument("--output-dir", help="Output directory")
    watch_parser.set_defaults(func=cmd_watch)

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model checkpoint")
    eval_parser.add_argument("--checkpoint", help="Checkpoint to evaluate")
    eval_parser.set_defaults(func=cmd_evaluate)

    # config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    config_parser.add_argument("--json", action="store_true", help="Output as JSON")
    config_parser.set_defaults(func=cmd_config)

    # version command
    version_parser = subparsers.add_parser("version", help="Show version")
    version_parser.set_defaults(func=cmd_version)

    return parser


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging_from_args(
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
    )

    if not args.command:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
