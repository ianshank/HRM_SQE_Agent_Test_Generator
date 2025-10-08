"""
CLI Interface for Drop Folder System.

Provides command-line interface for:
- Setting up folder structure
- Processing files manually
- Starting watch service
- Processing specific files
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from hrm_eval.utils.logging_utils import setup_logging
from .processor import DropFolderProcessor
from .watcher import DropFolderWatcher

logger = logging.getLogger(__name__)


def setup_folders(base_path: Optional[Path] = None):
    """
    Setup drop folder directory structure.
    
    Args:
        base_path: Base path for drop folders (default: ./drop_folder)
    """
    if base_path is None:
        base_path = Path.cwd() / "drop_folder"
    
    logger.info(f"Setting up drop folder structure at: {base_path}")
    
    # Create directories
    directories = [
        base_path / "input",
        base_path / "processing",
        base_path / "output",
        base_path / "archive",
        base_path / "errors",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {directory}")
    
    # Create sample requirement file
    sample_file = base_path / "input" / "sample_requirement.txt"
    if not sample_file.exists():
        sample_content = """Epic: User Authentication System

User Story 1: Login with Email
As a user, I want to log in with my email and password
So that I can access my account securely

AC: User must provide valid email format
AC: Password must be at least 8 characters
AC: System displays error message for invalid credentials
AC: Successful login redirects to dashboard

User Story 2: Password Reset
As a user, I want to reset my password if I forget it
So that I can regain access to my account

AC: User receives reset link via email
AC: Reset link expires after 24 hours
AC: New password must meet security requirements
"""
        sample_file.write_text(sample_content)
        logger.info(f"Created sample file: {sample_file}")
    
    # Create README
    readme_file = base_path / "README.md"
    if not readme_file.exists():
        readme_content = """# Drop Folder for Automated Test Generation

## Usage

1. Drop requirement files (.txt or .md) into the `input/` folder
2. Run processing:
   - **Manual**: `python -m hrm_eval.drop_folder process`
   - **Watch Mode**: `python -m hrm_eval.drop_folder watch`

## Directory Structure

- `input/` - Drop requirement files here
- `processing/` - Files being processed (temporary)
- `output/` - Generated test cases (organized by timestamp)
- `archive/` - Successfully processed files
- `errors/` - Failed files with error logs

## Output Format

Each processed file creates a timestamped directory in `output/`:
```
output/20251008_143022_auth_req/
├── test_cases.json          # Structured JSON
├── test_cases.md            # Readable Markdown
├── generation_report.md     # Processing metadata
└── metadata.json            # Processing info
```

## Requirement File Format

Requirements can be written in natural language. The system will detect:
- Epic titles (first line or "Epic:" prefix)
- User stories ("As a...", "US:", numbered items)
- Acceptance criteria ("AC:", "Given/When/Then", bullet points)

See `sample_requirement.txt` for an example.
"""
        readme_file.write_text(readme_content)
        logger.info(f"Created README: {readme_file}")
    
    print(f"\n✓ Drop folder structure created at: {base_path}")
    print(f"\n  To get started:")
    print(f"  1. Check the sample file: {sample_file}")
    print(f"  2. Process it: python -m hrm_eval.drop_folder process")
    print(f"  3. Or start watching: python -m hrm_eval.drop_folder watch\n")


def process_all(config_path: Optional[str] = None):
    """
    Process all pending files in input directory.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("Starting batch processing")
    
    try:
        processor = DropFolderProcessor(config_path)
        results = processor.process_all_pending()
        
        if not results:
            print("No files to process")
            return
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"\n{'='*60}")
        print(f"Processing Complete")
        print(f"{'='*60}")
        print(f"Total Files: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")
        
        for result in results:
            status = "✓" if result.success else "✗"
            filepath = Path(result.input_file).name
            print(f"{status} {filepath}")
            if result.success:
                print(f"   Output: {result.output_dir}")
                print(f"   Tests: {result.test_count}")
                print(f"   Time: {result.processing_time:.2f}s")
            else:
                print(f"   Error: {result.error_message}")
            print()
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


def process_file(filepath: str, config_path: Optional[str] = None):
    """
    Process a specific requirements file.
    
    Args:
        filepath: Path to requirements file
        config_path: Path to configuration file
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    logger.info(f"Processing single file: {filepath}")
    
    try:
        processor = DropFolderProcessor(config_path)
        result = processor.process_file(filepath)
        
        if result.success:
            print(f"\n✓ Successfully processed: {filepath.name}")
            print(f"  Output directory: {result.output_dir}")
            print(f"  Test cases generated: {result.test_count}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            if result.rag_enabled:
                print(f"  RAG examples used: {result.retrieved_examples}")
            print()
        else:
            print(f"\n✗ Failed to process: {filepath.name}")
            print(f"  Error: {result.error_message}\n")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"File processing failed: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


def watch_folder(interval: int = 5, config_path: Optional[str] = None):
    """
    Start watching input directory for new files.
    
    Args:
        interval: Polling interval in seconds
        config_path: Path to configuration file
    """
    logger.info("Starting watch mode")
    
    try:
        watcher = DropFolderWatcher(config_path=config_path)
        watcher.watch()
    except KeyboardInterrupt:
        logger.info("Watch mode stopped by user")
        print("\nWatch mode stopped")
    except Exception as e:
        logger.error(f"Watch mode failed: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Drop Folder System for Automated Test Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup folder structure
  python -m hrm_eval.drop_folder setup

  # Process all pending files
  python -m hrm_eval.drop_folder process

  # Process specific file
  python -m hrm_eval.drop_folder process-file input/requirements.txt

  # Start watching for new files
  python -m hrm_eval.drop_folder watch

For more information, see the documentation.
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file',
        default=None
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup drop folder structure')
    setup_parser.add_argument(
        '--path',
        type=str,
        help='Base path for drop folders (default: ./drop_folder)',
        default=None
    )
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process all pending files')
    
    # Process-file command
    process_file_parser = subparsers.add_parser('process-file', help='Process specific file')
    process_file_parser.add_argument(
        'filepath',
        type=str,
        help='Path to requirements file'
    )
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch folder for new files')
    watch_parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Polling interval in seconds (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'setup':
            base_path = Path(args.path) if args.path else None
            setup_folders(base_path)
        
        elif args.command == 'process':
            process_all(args.config)
        
        elif args.command == 'process-file':
            process_file(args.filepath, args.config)
        
        elif args.command == 'watch':
            watch_folder(args.interval, args.config)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
