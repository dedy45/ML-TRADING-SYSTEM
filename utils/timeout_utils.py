# Timeout protection utilities untuk mencegah hang
# Kompatibel dengan Windows dan Anaconda environment

import os
import sys
import time
import signal
import threading
import asyncio
import functools
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    """Custom timeout exception"""
    pass

class TimeoutManager:
    """Robust timeout manager untuk berbagai platform"""
    
    def __init__(self):
        self.is_windows = os.name == 'nt'
        self.active_timers = []
        
    @contextmanager
    def timeout(self, seconds: int, description: str = "Operation"):
        """Context manager untuk timeout protection"""
        if seconds <= 0:
            yield
            return
            
        start_time = time.time()
        timeout_occurred = threading.Event()
        
        def timeout_handler():
            timeout_occurred.set()
            
        timer = threading.Timer(seconds, timeout_handler)
        timer.start()
        self.active_timers.append(timer)
        
        try:
            yield timeout_occurred
            
            if timeout_occurred.is_set():
                elapsed = time.time() - start_time
                raise TimeoutException(f"{description} timed out after {elapsed:.1f} seconds (limit: {seconds}s)")
                
        finally:
            timer.cancel()
            if timer in self.active_timers:
                self.active_timers.remove(timer)
    
    def run_with_timeout(self, func: Callable, timeout_seconds: int, *args, **kwargs) -> Any:
        """Execute function dengan timeout menggunakan ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return result
            except FutureTimeoutError:
                logger.warning(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
                raise TimeoutException(f"Function {func.__name__} timed out")
    
    def cleanup_active_timers(self):
        """Cancel all active timers"""
        for timer in self.active_timers:
            timer.cancel()
        self.active_timers.clear()

# Global timeout manager instance
timeout_manager = TimeoutManager()

def timeout_decorator(seconds: int, description: str = None):
    """Decorator untuk timeout protection"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            desc = description or f"Function {func.__name__}"
            return timeout_manager.run_with_timeout(func, seconds, *args, **kwargs)
        return wrapper
    return decorator

@contextmanager
def safe_timeout(seconds: int, description: str = "Operation"):
    """Safe timeout context manager"""
    try:
        with timeout_manager.timeout(seconds, description) as timeout_event:
            yield timeout_event
    except TimeoutException as e:
        logger.error(f"Timeout occurred: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in timeout context: {e}")
        raise

class ProgressMonitor:
    """Monitor progress dan detect hang conditions"""
    
    def __init__(self, max_idle_time: int = 300):  # 5 minutes default
        self.max_idle_time = max_idle_time
        self.last_activity = time.time()
        self.is_active = False
        self._monitor_thread = None
        
    def activity(self, description: str = None):
        """Register activity to prevent timeout"""
        self.last_activity = time.time()
        if description:
            logger.debug(f"Activity: {description}")
    
    def start_monitoring(self, callback: Callable = None):
        """Start monitoring for hang detection"""
        self.is_active = True
        
        def monitor():
            while self.is_active:
                idle_time = time.time() - self.last_activity
                if idle_time > self.max_idle_time:
                    logger.warning(f"Hang detected: no activity for {idle_time:.1f} seconds")
                    if callback:
                        callback(idle_time)
                    break
                time.sleep(10)  # Check every 10 seconds
                
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)

class ExecutionGuard:
    """Guard execution dengan multiple protection layers"""
    
    def __init__(self, max_execution_time: int = 1800):
        self.max_execution_time = max_execution_time
        self.start_time = None
        self.progress_monitor = ProgressMonitor()
        
    def __enter__(self):
        self.start_time = time.time()
        self.progress_monitor.start_monitoring(self._handle_hang)
        logger.info(f"Execution guard started (max time: {self.max_execution_time}s)")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_monitor.stop_monitoring()
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Execution guard completed (elapsed: {elapsed:.1f}s)")
        
    def check_timeout(self):
        """Check if execution should timeout"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_execution_time:
                raise TimeoutException(f"Execution timeout after {elapsed:.1f} seconds")
                
    def activity(self, description: str = None):
        """Register activity"""
        self.progress_monitor.activity(description)
        self.check_timeout()
        
    def _handle_hang(self, idle_time: float):
        """Handle hang detection"""
        logger.error(f"Hang detected after {idle_time:.1f} seconds of inactivity")
        # Could implement additional hang recovery here

# Async utilities untuk non-blocking operations
class AsyncRunner:
    """Run operations asynchronously dengan timeout"""
    
    @staticmethod
    async def run_with_timeout(coro, timeout_seconds: int):
        """Run coroutine with timeout"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutException(f"Async operation timed out after {timeout_seconds} seconds")
    
    @staticmethod
    def run_sync_with_timeout(func, timeout_seconds: int, *args, **kwargs):
        """Run sync function in async context with timeout"""
        async def async_wrapper():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
        
        return asyncio.run(AsyncRunner.run_with_timeout(async_wrapper(), timeout_seconds))

# Memory management utilities
def check_memory_usage():
    """Check current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': memory_percent
        }
    except ImportError:
        return {"error": "psutil not available"}

def force_garbage_collection():
    """Force garbage collection"""
    import gc
    collected = gc.collect()
    logger.debug(f"Garbage collection freed {collected} objects")
    return collected

# Retry utilities
def retry_with_timeout(max_retries: int = 3, timeout_per_attempt: int = 60):
    """Decorator untuk retry dengan timeout per attempt"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Attempt {attempt + 1}/{max_retries} for {func.__name__}")
                    return timeout_manager.run_with_timeout(
                        func, timeout_per_attempt, *args, **kwargs
                    )
                except (TimeoutException, Exception) as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Brief pause between retries
                        
            logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
            
        return wrapper
    return decorator

# Export utilities
__all__ = [
    'TimeoutException',
    'TimeoutManager', 
    'timeout_manager',
    'timeout_decorator',
    'safe_timeout',
    'ProgressMonitor',
    'ExecutionGuard',
    'AsyncRunner',
    'check_memory_usage',
    'force_garbage_collection',
    'retry_with_timeout'
]
