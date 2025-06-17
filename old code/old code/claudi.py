# ===== ENHANCED ERROR HANDLING SYSTEM =====

import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager
from datetime import datetime
import sys
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum
import json

# ===== ERROR CLASSIFICATION =====

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL_API = "external_api"
    DATABASE = "database"
    FILE_OPERATION = "file_operation"
    CACHE = "cache"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RESOURCE_LIMIT = "resource_limit"
    CONFIGURATION = "configuration"

@dataclass
class ErrorContext:
    """Comprehensive error context for debugging and monitoring"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    user_message: str
    technical_details: str
    stack_trace: Optional[str] = None
    user_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3

# ===== CUSTOM EXCEPTIONS =====

class BaseAppException(Exception):
    """Base exception for all application errors"""
    def __init__(self, message: str, category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Dict[str, Any] = None, technical_details: str = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.technical_details = technical_details or str(self)
        self.timestamp = datetime.now()

class ValidationError(BaseAppException):
    """File validation, input validation errors"""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.LOW, context)

class ProcessingError(BaseAppException):
    """AI processing, text processing errors"""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Dict[str, Any] = None):
        super().__init__(message, ErrorCategory.PROCESSING, severity, context)

class ExternalAPIError(BaseAppException):
    """OpenAI API, external service errors"""
    def __init__(self, message: str, api_name: str, status_code: int = None, context: Dict[str, Any] = None):
        context = context or {}
        context.update({"api_name": api_name, "status_code": status_code})
        super().__init__(message, ErrorCategory.EXTERNAL_API, ErrorSeverity.HIGH, context)

class DatabaseError(BaseAppException):
    """Database connection, query errors"""
    def __init__(self, message: str, operation: str, context: Dict[str, Any] = None):
        context = context or {}
        context.update({"db_operation": operation})
        super().__init__(message, ErrorCategory.DATABASE, ErrorSeverity.HIGH, context)

class FileOperationError(BaseAppException):
    """File reading, writing, parsing errors"""
    def __init__(self, message: str, filename: str, operation: str, context: Dict[str, Any] = None):
        context = context or {}
        context.update({"filename": filename, "file_operation": operation})
        super().__init__(message, ErrorCategory.FILE_OPERATION, ErrorSeverity.MEDIUM, context)

class ResourceLimitError(BaseAppException):
    """Memory, timeout, rate limit errors"""
    def __init__(self, message: str, resource_type: str, limit_value: Any = None, context: Dict[str, Any] = None):
        context = context or {}
        context.update({"resource_type": resource_type, "limit": limit_value})
        super().__init__(message, ErrorCategory.RESOURCE_LIMIT, ErrorSeverity.HIGH, context)

class ConfigurationError(BaseAppException):
    """Configuration, environment errors"""
    def __init__(self, message: str, config_key: str = None, context: Dict[str, Any] = None):
        context = context or {}
        if config_key:
            context.update({"config_key": config_key})
        super().__init__(message, ErrorCategory.CONFIGURATION, ErrorSeverity.CRITICAL, context)

# ===== ERROR HANDLER REGISTRY =====

class ErrorHandlerRegistry:
    """Registry for error handling strategies"""
    
    def __init__(self):
        self._handlers: Dict[ErrorCategory, List[Callable]] = {}
        self._fallback_handlers: List[Callable] = []
        
    def register_handler(self, category: ErrorCategory, handler: Callable):
        """Register error handler for specific category"""
        if category not in self._handlers:
            self._handlers[category] = []
        self._handlers[category].append(handler)
    
    def register_fallback(self, handler: Callable):
        """Register fallback handler for unhandled errors"""
        self._fallback_handlers.append(handler)
    
    def handle_error(self, error: BaseAppException, context: ErrorContext) -> bool:
        """Handle error using registered handlers"""
        handlers = self._handlers.get(error.category, [])
        
        for handler in handlers:
            try:
                if handler(error, context):
                    return True
            except Exception as e:
                logger.error(f"Error handler failed: {e}")
        
        # Try fallback handlers
        for handler in self._fallback_handlers:
            try:
                if handler(error, context):
                    return True
            except Exception as e:
                logger.error(f"Fallback handler failed: {e}")
        
        return False

# ===== ERROR MONITORING =====

class ErrorMonitor:
    """Error monitoring and alerting system"""
    
    def __init__(self, max_errors_per_minute: int = 10):
        self.error_log: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        self.max_errors_per_minute = max_errors_per_minute
        self._lock = threading.Lock()
    
    def log_error(self, error_context: ErrorContext):
        """Log error for monitoring"""
        with self._lock:
            self.error_log.append(error_context)
            key = f"{error_context.category.value}:{error_context.component}"
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            
            # Check for error spikes
            self._check_error_spike(error_context)
    
    def _check_error_spike(self, error_context: ErrorContext):
        """Check for error spikes and trigger alerts"""
        recent_errors = [e for e in self.error_log[-50:] 
                        if (datetime.now() - e.timestamp).seconds < 60]
        
        if len(recent_errors) > self.max_errors_per_minute:
            logger.critical(f"Error spike detected: {len(recent_errors)} errors in last minute")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for dashboard"""
        with self._lock:
            return {
                "total_errors": len(self.error_log),
                "error_by_category": {cat.value: sum(1 for e in self.error_log if e.category == cat) 
                                    for cat in ErrorCategory},
                "error_by_severity": {sev.value: sum(1 for e in self.error_log if e.severity == sev) 
                                    for sev in ErrorSeverity},
                "recent_errors": [e for e in self.error_log[-10:]]
            }

# ===== RETRY MECHANISM =====

class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 exponential_backoff: bool = True, jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter

def with_retry(retry_config: RetryConfig = None, 
               retryable_exceptions: tuple = (Exception,),
               non_retryable_exceptions: tuple = (ValidationError, ConfigurationError)):
    """Decorator for automatic retry logic"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except non_retryable_exceptions as e:
                    # Don't retry these exceptions
                    raise e
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_attempts - 1:
                        # Last attempt, raise the exception
                        break
                    
                    # Calculate delay
                    delay = retry_config.base_delay
                    if retry_config.exponential_backoff:
                        delay *= (2 ** attempt)
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + 0.5 * random.random())
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator

# ===== CIRCUIT BREAKER =====

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern for external dependencies"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, 
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise ExternalAPIError(
                        "Circuit breaker is OPEN - service unavailable",
                        api_name=func.__name__,
                        context={"circuit_breaker_state": self.state.value}
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.timeout)
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

# ===== COMPREHENSIVE ERROR HANDLER =====

class ComprehensiveErrorHandler:
    """Main error handling orchestrator"""
    
    def __init__(self):
        self.registry = ErrorHandlerRegistry()
        self.monitor = ErrorMonitor()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default error handlers"""
        
        # Validation error handler
        def handle_validation_error(error: ValidationError, context: ErrorContext) -> bool:
            if hasattr(st, 'error'):  # Streamlit context
                st.error(f"❌ {error.message}")
            logger.warning(f"Validation error: {error.message}")
            return True
        
        # Processing error handler
        def handle_processing_error(error: ProcessingError, context: ErrorContext) -> bool:
            if hasattr(st, 'error'):  # Streamlit context
                if error.severity == ErrorSeverity.HIGH:
                    st.error(f"🚨 Критическая ошибка обработки: {error.message}")
                else:
                    st.warning(f"⚠️ Ошибка обработки: {error.message}")
            logger.error(f"Processing error: {error.message}")
            return True
        
        # External API error handler
        def handle_api_error(error: ExternalAPIError, context: ErrorContext) -> bool:
            if hasattr(st, 'error'):  # Streamlit context
                st.error(f"🌐 Ошибка внешнего сервиса ({error.context.get('api_name', 'Unknown')}): {error.message}")
            logger.error(f"API error: {error.message}", extra=error.context)
            return True
        
        # Database error handler
        def handle_db_error(error: DatabaseError, context: ErrorContext) -> bool:
            if hasattr(st, 'error'):  # Streamlit context
                st.error("🗄️ Ошибка базы данных. Попробуйте позже.")
            logger.error(f"Database error: {error.message}", extra=error.context)
            return True
        
        # File operation error handler
        def handle_file_error(error: FileOperationError, context: ErrorContext) -> bool:
            if hasattr(st, 'error'):  # Streamlit context
                st.error(f"📄 Ошибка работы с файлом '{error.context.get('filename', 'Unknown')}': {error.message}")
            logger.error(f"File error: {error.message}", extra=error.context)
            return True
        
        # Resource limit error handler
        def handle_resource_error(error: ResourceLimitError, context: ErrorContext) -> bool:
            if hasattr(st, 'error'):  # Streamlit context
                st.error(f"⚡ Превышен лимит ресурсов ({error.context.get('resource_type', 'Unknown')}): {error.message}")
            logger.error(f"Resource limit error: {error.message}", extra=error.context)
            return True
        
        # Configuration error handler
        def handle_config_error(error: ConfigurationError, context: ErrorContext) -> bool:
            if hasattr(st, 'error'):  # Streamlit context
                st.error("⚙️ Ошибка конфигурации. Обратитесь к администратору.")
            logger.critical(f"Configuration error: {error.message}", extra=error.context)
            return True
        
        # Register handlers
        self.registry.register_handler(ErrorCategory.VALIDATION, handle_validation_error)
        self.registry.register_handler(ErrorCategory.PROCESSING, handle_processing_error)
        self.registry.register_handler(ErrorCategory.EXTERNAL_API, handle_api_error)
        self.registry.register_handler(ErrorCategory.DATABASE, handle_db_error)
        self.registry.register_handler(ErrorCategory.FILE_OPERATION, handle_file_error)
        self.registry.register_handler(ErrorCategory.RESOURCE_LIMIT, handle_resource_error)
        self.registry.register_handler(ErrorCategory.CONFIGURATION, handle_config_error)
    
    def handle_exception(self, component: str, operation: str, exception: Exception, 
                        user_data: Dict[str, Any] = None) -> ErrorContext:
        """Main exception handling entry point"""
        
        # Create error context
        error_context = ErrorContext(
            error_id=f"{component}_{operation}_{int(time.time())}",
            timestamp=datetime.now(),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PROCESSING,
            component=component,
            operation=operation,
            user_message="Произошла ошибка",
            technical_details=str(exception),
            stack_trace=traceback.format_exc(),
            user_data=user_data or {},
            system_state=self._get_system_state()
        )
        
        # Determine error type and update context
        if isinstance(exception, BaseAppException):
            error_context.severity = exception.severity
            error_context.category = exception.category
            error_context.user_message = exception.message
            error_context.technical_details = exception.technical_details
        else:
            # Map common exceptions to categories
            error_context.category = self._categorize_exception(exception)
            error_context.user_message = self._get_user_friendly_message(exception)
        
        # Log error
        logger.error(f"Error in {component}.{operation}: {exception}", 
                    exc_info=True, extra={"error_context": error_context})
        
        # Monitor error
        self.monitor.log_error(error_context)
        
        # Handle error
        try:
            if isinstance(exception, BaseAppException):
                self.registry.handle_error(exception, error_context)
            else:
                # Create BaseAppException wrapper
                wrapped_error = BaseAppException(
                    str(exception), error_context.category, error_context.severity
                )
                self.registry.handle_error(wrapped_error, error_context)
        except Exception as handler_error:
            logger.error(f"Error handler failed: {handler_error}")
            if hasattr(st, 'error'):
                st.error("Произошла критическая ошибка системы.")
        
        return error_context
    
    def _categorize_exception(self, exception: Exception) -> ErrorCategory:
        """Categorize unknown exceptions"""
        exception_type = type(exception).__name__
        
        if any(keyword in str(exception).lower() for keyword in ['connection', 'network', 'timeout']):
            return ErrorCategory.NETWORK
        elif any(keyword in str(exception).lower() for keyword in ['database', 'sql', 'connection']):
            return ErrorCategory.DATABASE
        elif any(keyword in str(exception).lower() for keyword in ['file', 'permission', 'not found']):
            return ErrorCategory.FILE_OPERATION
        elif any(keyword in str(exception).lower() for keyword in ['memory', 'limit', 'quota']):
            return ErrorCategory.RESOURCE_LIMIT
        elif any(keyword in str(exception).lower() for keyword in ['auth', 'token', 'credential']):
            return ErrorCategory.AUTHENTICATION
        else:
            return ErrorCategory.PROCESSING
    
    def _get_user_friendly_message(self, exception: Exception) -> str:
        """Generate user-friendly error messages"""
        exception_str = str(exception).lower()
        
        if 'connection' in exception_str:
            return "Проблема с подключением к сервису"
        elif 'timeout' in exception_str:
            return "Операция заняла слишком много времени"
        elif 'permission' in exception_str:
            return "Недостаточно прав для выполнения операции"
        elif 'file not found' in exception_str:
            return "Файл не найден"
        elif 'memory' in exception_str:
            return "Недостаточно памяти для выполнения операции"
        else:
            return "Произошла техническая ошибка"
    
    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state for debugging"""
        return {
            "timestamp": datetime.now().isoformat(),
            "thread_count": threading.active_count(),
            "memory_usage": "N/A",  # Would need psutil for real memory info
        }
    
    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]

# ===== DECORATORS FOR EASY INTEGRATION =====

# Global error handler instance
error_handler = ComprehensiveErrorHandler()

def handle_errors(component: str, operation: str = None):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = error_handler.handle_exception(
                    component=component,
                    operation=op_name,
                    exception=e,
                    user_data={"args": str(args)[:100], "kwargs": str(kwargs)[:100]}
                )
                
                # Re-raise if it's a critical error
                if isinstance(e, BaseAppException) and e.severity == ErrorSeverity.CRITICAL:
                    raise e
                
                # Return None or appropriate default for non-critical errors
                return None
        return wrapper
    return decorator

@contextmanager
def error_context(component: str, operation: str, user_data: Dict[str, Any] = None):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        error_handler.handle_exception(component, operation, e, user_data)
        raise

# ===== INTEGRATION EXAMPLES =====

# Example usage with your existing classes:

class EnhancedFileProcessor(FileProcessor):
    @handle_errors("FileProcessor", "validate_file")
    def validate_file(self, file) -> Tuple[bool, str]:
        try:
            if not file:
                raise ValidationError("Файл не выбран")
            
            if file.size > self.max_file_size_bytes:
                raise ValidationError(
                    f"Файл слишком большой. Максимальный размер: {self.max_file_size_bytes // (1024*1024)}MB",
                    context={"file_size": file.size, "max_size": self.max_file_size_bytes}
                )
            
            return True, ""
        except ValidationError:
            raise
        except Exception as e:
            raise FileOperationError(
                f"Ошибка при валидации файла: {str(e)}",
                filename=getattr(file, 'name', 'Unknown'),
                operation="validation"
            )

class EnhancedAIService(AIService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit_breaker = error_handler.get_circuit_breaker("openai_api")
    
    @handle_errors("AIService", "analyze_text")
    @with_retry(RetryConfig(max_attempts=3, base_delay=2.0))
    async def analyze_text_for_ai(self, text: str) -> ProcessingResult:
        try:
            return await self.circuit_breaker.call(self._analyze_text_internal, text)
        except ExternalAPIError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Ошибка анализа текста: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context={"text_length": len(text)}
            )
    
    async def _analyze_text_internal(self, text: str) -> ProcessingResult:
        # Your existing analysis logic here
        # Wrap OpenAI API calls with proper error handling
        try:
            # ... existing code ...
            pass
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise ResourceLimitError(
                    "Превышен лимит запросов к API",
                    resource_type="api_rate_limit"
                )
            elif "authentication" in str(e).lower():
                raise ExternalAPIError(
                    "Ошибка аутентификации API",
                    api_name="OpenAI",
                    context={"error": str(e)}
                )
            else:
                raise ExternalAPIError(
                    f"Ошибка API: {str(e)}",
                    api_name="OpenAI"
                )

# Example Streamlit integration
def create_error_dashboard():
    """Create error monitoring dashboard in Streamlit"""
    st.subheader("🔍 Мониторинг ошибок")
    
    error_summary = error_handler.monitor.get_error_summary()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего ошибок", error_summary["total_errors"])
    with col2:
        critical_errors = error_summary["error_by_severity"].get("critical", 0)
        st.metric("Критических ошибок", critical_errors)
    with col3:
        recent_errors = len(error_summary["recent_errors"])
        st.metric("Недавних ошибок", recent_errors)
    
    # Error breakdown
    st.subheader("Категории ошибок")
    error_df = pd.DataFrame([
        {"Категория": cat, "Количество": count}
        for cat, count in error_summary["error_by_category"].items()
        if count > 0
    ])
    
    if not error_df.empty:
        st.bar_chart(error_df.set_index("Категория"))
    else:
        st.info("Ошибок не обнаружено")
    
    # Recent errors
    if error_summary["recent_errors"]:
        st.subheader("Последние ошибки")
        for error in error_summary["recent_errors"][-5:]:
            with st.expander(f"{error.component}.{error.operation} - {error.timestamp.strftime('%H:%M:%S')}"):
                st.write(f"**Уровень:** {error.severity.value}")
                st.write(f"**Категория:** {error.category.value}")
                st.write(f"**Сообщение:** {error.user_message}")
                if error.technical_details:
                    st.code(error.technical_details)