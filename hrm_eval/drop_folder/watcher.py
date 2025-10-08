"""
File Watcher Service for Drop Folder.

Monitors the input directory for new requirement files and triggers processing.
"""

import time
import logging
import signal
import sys
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime, timedelta

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("watchdog library not available, using polling mode")

from .processor import DropFolderProcessor

logger = logging.getLogger(__name__)


class RequirementsFileHandler(FileSystemEventHandler):
    """
    Event handler for file system changes.
    
    Monitors for new files and triggers processing after debounce period.
    """
    
    def __init__(
        self,
        processor: DropFolderProcessor,
        debounce_delay: float = 2.0,
        allowed_extensions: tuple = ('.txt', '.md')
    ):
        """
        Initialize file handler.
        
        Args:
            processor: DropFolderProcessor instance
            debounce_delay: Seconds to wait after last modification
            allowed_extensions: Tuple of allowed file extensions
        """
        super().__init__()
        self.processor = processor
        self.debounce_delay = debounce_delay
        self.allowed_extensions = allowed_extensions
        self.pending_files = {}  # filepath -> last_modified_time
        
        logger.info(f"File handler initialized with {debounce_delay}s debounce")
    
    def on_created(self, event: FileSystemEvent):
        """Handle file creation event."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        if filepath.suffix in self.allowed_extensions:
            logger.info(f"New file detected: {filepath.name}")
            self.pending_files[filepath] = datetime.now()
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification event."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        if filepath.suffix in self.allowed_extensions:
            logger.debug(f"File modified: {filepath.name}")
            self.pending_files[filepath] = datetime.now()
    
    def process_pending(self):
        """Process files that have been stable for debounce period."""
        now = datetime.now()
        to_process = []
        
        for filepath, last_modified in list(self.pending_files.items()):
            if now - last_modified >= timedelta(seconds=self.debounce_delay):
                if filepath.exists():
                    to_process.append(filepath)
                del self.pending_files[filepath]
        
        for filepath in to_process:
            logger.info(f"Processing stable file: {filepath.name}")
            try:
                result = self.processor.process_file(filepath)
                if result.success:
                    logger.info(f"Successfully processed: {filepath.name}")
                else:
                    logger.error(f"Failed to process: {filepath.name} - {result.error_message}")
            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}", exc_info=True)


class DropFolderWatcher:
    """
    Watches drop folder for new requirement files.
    
    Supports:
    - Continuous monitoring with watchdog
    - Polling mode as fallback
    - Batch processing mode
    - Graceful shutdown
    """
    
    def __init__(
        self,
        processor: Optional[DropFolderProcessor] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize watcher.
        
        Args:
            processor: DropFolderProcessor instance (creates new if None)
            config_path: Path to configuration file
        """
        logger.info("Initializing DropFolderWatcher")
        
        self.processor = processor or DropFolderProcessor(config_path)
        self.config = self.processor.config
        
        self.input_dir = self.processor.input_dir
        self.watch_interval = self.config.get('watch_interval', 5)
        self.debounce_delay = self.config.get('debounce_delay', 2)
        
        self.observer = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Watcher initialized for directory: {self.input_dir}")
    
    def watch(self):
        """
        Start watching directory for changes.
        
        Uses watchdog if available, falls back to polling.
        """
        logger.info("Starting file watcher...")
        
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        if WATCHDOG_AVAILABLE:
            self._watch_with_watchdog()
        else:
            self._watch_with_polling()
    
    def _watch_with_watchdog(self):
        """Watch using watchdog library (event-driven)."""
        logger.info("Using watchdog for file monitoring")
        
        allowed_extensions = tuple(self.config.get('file_extensions', ['.txt', '.md']))
        event_handler = RequirementsFileHandler(
            processor=self.processor,
            debounce_delay=self.debounce_delay,
            allowed_extensions=allowed_extensions
        )
        
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.input_dir), recursive=False)
        self.observer.start()
        self.running = True
        
        logger.info(f"Monitoring {self.input_dir} for new files...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while self.running:
                time.sleep(1)
                # Process pending files periodically
                if hasattr(event_handler, 'process_pending'):
                    event_handler.process_pending()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.stop()
    
    def _watch_with_polling(self):
        """Watch using simple polling (fallback)."""
        logger.info(f"Using polling mode with {self.watch_interval}s interval")
        logger.warning("Install 'watchdog' package for better performance: pip install watchdog")
        
        processed_files = set()
        self.running = True
        
        logger.info(f"Monitoring {self.input_dir} for new files...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while self.running:
                # Find new files
                current_files = set()
                for ext in self.config.get('file_extensions', ['.txt', '.md']):
                    current_files.update(self.input_dir.glob(f"*{ext}"))
                
                new_files = current_files - processed_files
                
                if new_files:
                    logger.info(f"Found {len(new_files)} new files")
                    for filepath in new_files:
                        logger.info(f"Processing: {filepath.name}")
                        try:
                            result = self.processor.process_file(filepath)
                            if result.success:
                                logger.info(f"Successfully processed: {filepath.name}")
                                processed_files.add(filepath)
                            else:
                                logger.error(f"Failed: {filepath.name} - {result.error_message}")
                        except Exception as e:
                            logger.error(f"Error processing {filepath.name}: {e}", exc_info=True)
                
                time.sleep(self.watch_interval)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            self.running = False
    
    def run_batch(self):
        """
        Process all pending files once (batch mode).
        
        Does not watch continuously.
        """
        logger.info("Running batch processing mode")
        
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        results = self.processor.process_all_pending()
        
        if not results:
            logger.info("No files to process")
            return
        
        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_time = sum(r.processing_time for r in results)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Batch Processing Complete")
        logger.info("=" * 60)
        logger.info(f"Total Files: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info("=" * 60)
        
        for result in results:
            status = "✓" if result.success else "✗"
            logger.info(f"{status} {Path(result.input_file).name} - {result.processing_time:.2f}s")
    
    def stop(self):
        """Stop watching and cleanup."""
        logger.info("Stopping watcher...")
        self.running = False
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
            logger.info("Observer stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
