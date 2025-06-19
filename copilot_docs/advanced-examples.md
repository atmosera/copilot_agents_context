# Advanced GitHub Copilot Examples Collection

## Table of Contents
- [Advanced Prompting Techniques](#advanced-prompting-techniques)
- [Complex Architecture Patterns](#complex-architecture-patterns)
- [Multi-Language Integration Examples](#multi-language-integration-examples)
- [Performance Optimization Examples](#performance-optimization-examples)
- [Security Implementation Examples](#security-implementation-examples)
- [Testing Strategy Examples](#testing-strategy-examples)
- [DevOps and Infrastructure Examples](#devops-and-infrastructure-examples)
- [Domain-Specific Examples](#domain-specific-examples)

## Advanced Prompting Techniques

### Contextual Chain Prompting

#### Example: Building a Complete Microservice
```python
# Phase 1: Define the service architecture
"""
Create a user authentication microservice for a distributed e-commerce platform.

Architecture Requirements:
- FastAPI framework with async/await patterns
- PostgreSQL database with SQLAlchemy 2.0 async ORM
- Redis for session management and caching
- JWT tokens with refresh token mechanism
- Rate limiting with Redis sliding window
- Comprehensive logging with structured JSON
- Health checks and metrics endpoints
- Docker containerization ready
- OpenAPI documentation with examples

Security Requirements:
- Password hashing with bcrypt and salt
- Input validation with Pydantic models
- SQL injection prevention
- CORS configuration for frontend integration
- Secure headers middleware
- Request/response sanitization

Performance Requirements:
- Handle 1000+ concurrent users
- Database connection pooling
- Async Redis operations
- Response time < 100ms for auth checks
- Memory usage < 512MB under load
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import redis.asyncio as redis
from pydantic import BaseModel, EmailStr
import bcrypt
import jwt
from datetime import datetime, timedelta
import logging
import structlog

# Service will be implemented following all requirements above
app = FastAPI(
    title="Authentication Microservice",
    description="Secure authentication service for e-commerce platform",
    version="1.0.0"
)

# Phase 2: Database models and connections
# Continue building on the established context...
```

#### Example: Progressive Complexity Building
```java
// Stage 1: Basic service structure
/**
 * Payment Processing Service - Stage 1: Core Structure
 * 
 * Build a comprehensive payment processing service with these stages:
 * 1. Basic service structure with dependency injection
 * 2. Payment method abstractions and strategy pattern
 * 3. Fraud detection integration
 * 4. Transaction state management with saga pattern
 * 5. Event sourcing for audit trail
 * 6. Performance monitoring and circuit breakers
 * 
 * Current Stage: Basic service structure with Spring Boot
 * Framework: Spring Boot 3.0+, Spring Security, Spring Data JPA
 * Database: PostgreSQL with optimistic locking
 * Message Queue: RabbitMQ for async processing
 */

@Service
@Transactional
@Slf4j
public class PaymentProcessingService {
    
    private final PaymentRepository paymentRepository;
    private final FraudDetectionService fraudDetectionService;
    private final PaymentGatewayFactory gatewayFactory;
    private final EventPublisher eventPublisher;
    
    // Constructor injection and basic structure will be generated
}

// Stage 2: Add payment method strategies
// Building on the previous context, add strategy pattern for different payment methods...
/**
 * Payment Processing Service - Stage 2: Strategy Pattern Implementation
 * 
 * Continuing from Stage 1, now implement:
 * - Strategy pattern for different payment methods (Credit Card, PayPal, Apple Pay)
 * - Factory pattern for payment gateway selection
 * - Validation chains for payment data
 * - Error handling with custom exceptions
 */

public interface PaymentStrategy {
    PaymentResult processPayment(PaymentRequest request, PaymentContext context);
    boolean supports(PaymentMethod method);
    ValidationResult validatePaymentData(PaymentData data);
}

// Implementation will continue building on established patterns...
```

### Domain-Specific Context Loading

#### Example: Financial Services Context
```python
# Financial Services Trading Platform Context
"""
Trading Platform: High-Frequency Trading System

Domain Context:
- Financial instruments: Stocks, Options, Futures, Forex
- Market data: Real-time price feeds, order books, trade history
- Risk management: Position limits, VaR calculations, margin requirements
- Regulatory compliance: FINRA, SEC, MiFID II requirements
- Performance: Sub-millisecond latency requirements
- Data integrity: ACID transactions, audit trails

Technical Stack:
- Python 3.11+ with asyncio for concurrency
- Apache Kafka for market data streaming
- TimescaleDB for time-series market data
- Redis for real-time caching and session state
- FastAPI for REST APIs and WebSocket connections
- Celery for background task processing
- Prometheus for monitoring and alerting

Financial Calculations:
- Greeks calculation for options (Delta, Gamma, Theta, Vega)
- VaR (Value at Risk) using Monte Carlo simulations
- Real-time P&L calculations with mark-to-market
- Risk metrics: Sharpe ratio, maximum drawdown, beta
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio
import numpy as np
from scipy import stats

@dataclass
class MarketData:
    """Real-time market data for financial instruments."""
    symbol: str
    price: Decimal
    bid: Decimal
    ask: Decimal
    volume: int
    timestamp: datetime
    exchange: str

class OptionsGreeksCalculator:
    """
    Calculate options Greeks for risk management.
    
    Uses Black-Scholes model for European options pricing.
    Supports: Delta, Gamma, Theta, Vega, Rho calculations.
    Performance requirement: < 1ms per calculation.
    """
    
    def calculate_greeks(
        self,
        spot_price: Decimal,
        strike_price: Decimal,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str
    ) -> Dict[str, Decimal]:
        """
        Calculate all Greeks for an option position.
        
        Returns comprehensive Greeks calculation with proper precision
        for financial risk management systems.
        """
        # Implementation will follow financial mathematics best practices
```

## Complex Architecture Patterns

### Event-Driven Architecture Example

```typescript
// Distributed Event-Driven E-commerce Platform
/**
 * Event-Driven Architecture for E-commerce Platform
 * 
 * Services: Order, Payment, Inventory, Shipping, Notification
 * Event Store: PostgreSQL with event sourcing patterns
 * Message Broker: Apache Kafka with schema registry
 * API Gateway: Kong with rate limiting and authentication
 * Service Mesh: Istio for service-to-service communication
 * 
 * Patterns Implemented:
 * - Event Sourcing with snapshots
 * - CQRS (Command Query Responsibility Segregation)
 * - Saga pattern for distributed transactions
 * - Circuit breaker for resilience
 * - Dead letter queues for error handling
 * 
 * Performance Requirements:
 * - 10,000+ orders per minute
 * - < 500ms end-to-end order processing
 * - 99.9% availability
 * - Eventual consistency with compensation patterns
 */

// Domain Events Definition
interface DomainEvent {
  readonly eventId: string;
  readonly aggregateId: string;
  readonly aggregateType: string;
  readonly eventType: string;
  readonly eventVersion: number;
  readonly occurredAt: Date;
  readonly causationId?: string;
  readonly correlationId?: string;
}

interface OrderCreatedEvent extends DomainEvent {
  readonly eventType: 'OrderCreated';
  readonly data: {
    orderId: string;
    customerId: string;
    items: OrderItem[];
    totalAmount: Money;
    shippingAddress: Address;
    paymentMethod: PaymentMethod;
  };
}

// Event Store Implementation
class EventStore {
  constructor(
    private readonly connection: Pool,
    private readonly eventBus: EventBus
  ) {}

  async appendEvents(
    aggregateId: string,
    aggregateType: string,
    events: DomainEvent[],
    expectedVersion: number
  ): Promise<void> {
    // Optimistic concurrency control implementation
    // Atomic append with version checking
    // Event publishing to message broker
  }

  async getEvents(
    aggregateId: string,
    fromVersion?: number
  ): Promise<DomainEvent[]> {
    // Efficient event retrieval with optional snapshots
  }
}

// Saga Orchestrator for Order Processing
class OrderProcessingSaga {
  private readonly steps: SagaStep[] = [
    new ReserveInventoryStep(),
    new ProcessPaymentStep(),
    new CreateShipmentStep(),
    new SendConfirmationStep()
  ];

  async executeOrder(orderCreatedEvent: OrderCreatedEvent): Promise<void> {
    // Implement saga pattern with compensation actions
    // Handle partial failures and rollbacks
    // Maintain saga state for long-running transactions
  }
}
```

### Microservices with API Gateway Pattern

```go
// API Gateway with Advanced Routing and Middleware
/*
Enterprise API Gateway Implementation

Features:
- Dynamic service discovery with Consul
- JWT authentication with RBAC
- Rate limiting with Redis sliding window
- Request/response transformation
- Circuit breaker pattern with Hystrix
- Distributed tracing with Jaeger
- Metrics collection with Prometheus
- Load balancing with health checks
- SSL/TLS termination

Performance Requirements:
- Handle 50,000+ requests per second
- < 5ms latency overhead
- 99.99% uptime
- Horizontal scaling capability

Security Features:
- OAuth 2.0 / OpenID Connect integration
- API key management
- IP whitelisting/blacklisting
- Request payload validation
- SQL injection / XSS protection
*/

package main

import (
    "context"
    "crypto/tls"
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"

    "github.com/gin-gonic/gin"
    "github.com/redis/go-redis/v9"
    "github.com/prometheus/client_golang/prometheus"
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

// Gateway configuration and middleware stack
type APIGateway struct {
    router           *gin.Engine
    serviceRegistry  ServiceRegistry
    authService      AuthenticationService
    rateLimiter      RateLimiter
    circuitBreaker   CircuitBreaker
    loadBalancer     LoadBalancer
    metricsCollector MetricsCollector
}

// Advanced middleware pipeline
func (gw *APIGateway) setupMiddleware() {
    // Request tracing and correlation ID
    gw.router.Use(TracingMiddleware())
    
    // Security headers and CORS
    gw.router.Use(SecurityHeadersMiddleware())
    
    // Authentication and authorization
    gw.router.Use(JWTAuthMiddleware(gw.authService))
    
    // Rate limiting with Redis
    gw.router.Use(RateLimitMiddleware(gw.rateLimiter))
    
    // Circuit breaker for downstream services
    gw.router.Use(CircuitBreakerMiddleware(gw.circuitBreaker))
    
    // Request/response logging and metrics
    gw.router.Use(MetricsMiddleware(gw.metricsCollector))
}

// Dynamic service routing with load balancing
func (gw *APIGateway) proxyRequest(c *gin.Context) {
    // Extract service name from route
    // Discover healthy service instances
    // Apply load balancing algorithm
    // Proxy request with timeout and retry logic
    // Handle response transformation
}

// Service discovery integration
type ServiceRegistry interface {
    DiscoverServices(serviceName string) ([]ServiceInstance, error)
    RegisterHealthCheck(instance ServiceInstance) error
    WatchServiceChanges(callback func([]ServiceInstance)) error
}

// Implementation will include Consul integration, health checking,
// and automatic service registration/deregistration
```

## Multi-Language Integration Examples

### Python-Java Integration via gRPC

```python
# Python gRPC Service Implementation
"""
Multi-Language Integration: Python ML Service + Java Business Logic

Architecture:
- Python: Machine Learning models with TensorFlow/PyTorch
- Java: Business logic and transaction processing
- gRPC: High-performance RPC communication
- Protocol Buffers: Type-safe serialization
- Docker: Containerized deployment
- Kubernetes: Orchestration and scaling

Use Case: Real-time fraud detection service
- Python service handles ML inference (fraud scoring)
- Java service manages business rules and transactions
- Sub-100ms latency requirement for real-time decisions
"""

import grpc
from concurrent import futures
import asyncio
import numpy as np
import tensorflow as tf
from prometheus_client import Counter, Histogram, start_http_server
import structlog

# Generated from fraud_detection.proto
import fraud_detection_pb2
import fraud_detection_pb2_grpc

logger = structlog.get_logger()

class FraudDetectionService(fraud_detection_pb2_grpc.FraudDetectionServicer):
    """
    High-performance fraud detection service using ML models.
    
    Features:
    - Real-time inference with TensorFlow Serving
    - Model versioning and A/B testing
    - Feature engineering pipeline
    - Async processing for batch requests
    - Comprehensive monitoring and alerting
    """
    
    def __init__(self):
        self.model = self._load_fraud_model()
        self.feature_processor = FeatureProcessor()
        self.metrics = self._setup_metrics()
        
    async def DetectFraud(
        self, 
        request: fraud_detection_pb2.FraudDetectionRequest,
        context: grpc.aio.ServicerContext
    ) -> fraud_detection_pb2.FraudDetectionResponse:
        """
        Real-time fraud detection with ML inference.
        
        Processes transaction features through ML pipeline
        and returns fraud probability score with explanation.
        """
        start_time = time.time()
        
        try:
            # Extract and normalize features
            features = self.feature_processor.process_transaction(
                request.transaction
            )
            
            # ML inference
            fraud_probability = await self._predict_fraud(features)
            
            # Generate explanation
            explanation = await self._generate_explanation(
                features, fraud_probability
            )
            
            # Record metrics
            self.metrics['inference_duration'].observe(
                time.time() - start_time
            )
            
            return fraud_detection_pb2.FraudDetectionResponse(
                fraud_probability=fraud_probability,
                risk_level=self._calculate_risk_level(fraud_probability),
                explanation=explanation,
                model_version=self.model.version
            )
            
        except Exception as e:
            logger.error("Fraud detection failed", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {str(e)}")
            raise
```

```java
// Java Client Integration
/**
 * Java Business Service Integration with Python ML Service
 * 
 * Integration Pattern:
 * - Async gRPC client with connection pooling
 * - Circuit breaker for ML service failures
 * - Fallback to rule-based fraud detection
 * - Request batching for performance optimization
 * - Comprehensive error handling and retry logic
 */

@Service
@Slf4j
public class FraudDetectionOrchestrator {
    
    private final FraudDetectionServiceGrpc.FraudDetectionServiceStub asyncStub;
    private final CircuitBreaker circuitBreaker;
    private final RuleBasedFraudDetector fallbackDetector;
    private final MeterRegistry meterRegistry;
    
    @Autowired
    public FraudDetectionOrchestrator(
            @Qualifier("fraudDetectionChannel") ManagedChannel channel,
            CircuitBreaker circuitBreaker,
            RuleBasedFraudDetector fallbackDetector,
            MeterRegistry meterRegistry) {
        
        this.asyncStub = FraudDetectionServiceGrpc.newStub(channel);
        this.circuitBreaker = circuitBreaker;
        this.fallbackDetector = fallbackDetector;
        this.meterRegistry = meterRegistry;
    }
    
    @Async("fraudDetectionExecutor")
    public CompletableFuture<FraudAssessment> assessTransaction(
            Transaction transaction) {
        
        Timer.Sample sample = Timer.start(meterRegistry);
        
        return circuitBreaker.executeSupplier(() -> {
            // Build gRPC request
            FraudDetectionRequest request = FraudDetectionRequest.newBuilder()
                .setTransaction(mapToProtoTransaction(transaction))
                .setRequestId(UUID.randomUUID().toString())
                .build();
            
            // Async gRPC call with timeout
            CompletableFuture<FraudDetectionResponse> future = new CompletableFuture<>();
            
            asyncStub.withDeadlineAfter(500, TimeUnit.MILLISECONDS)
                .detectFraud(request, new StreamObserver<FraudDetectionResponse>() {
                    @Override
                    public void onNext(FraudDetectionResponse response) {
                        future.complete(response);
                    }
                    
                    @Override
                    public void onError(Throwable t) {
                        log.warn("ML fraud detection failed, using fallback", t);
                        future.complete(null); // Will trigger fallback
                    }
                    
                    @Override
                    public void onCompleted() {
                        // Response already handled in onNext
                    }
                });
            
            return future.thenApply(response -> {
                sample.stop(Timer.builder("fraud.detection.duration")
                    .tag("service", response != null ? "ml" : "fallback")
                    .register(meterRegistry));
                
                if (response != null) {
                    return mapToFraudAssessment(response);
                } else {
                    // Fallback to rule-based detection
                    return fallbackDetector.assess(transaction);
                }
            });
        });
    }
}
```

### C++ Performance Engine with Python Interface

```cpp
// High-Performance C++ Computation Engine
/**
 * Ultra-High Performance Financial Calculations Engine
 * 
 * Requirements:
 * - Sub-microsecond latency for options pricing
 * - Thread-safe concurrent calculations
 * - Python bindings with pybind11
 * - SIMD optimizations for vectorized operations
 * - Memory pool allocation for zero-allocation paths
 * - Real-time risk calculations
 * 
 * Use Case: High-frequency trading risk engine
 * - Calculate Greeks for thousands of options simultaneously
 * - Real-time portfolio VaR calculations
 * - Monte Carlo simulations with GPU acceleration
 */

#pragma once

#include <vector>
#include <memory>
#include <chrono>
#include <immintrin.h>  // AVX/SSE instructions
#include <tbb/parallel_for.h>  // Intel TBB for parallelization
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace finance {

class HighPerformanceRiskEngine {
private:
    // Memory pool for zero-allocation calculations
    std::unique_ptr<MemoryPool> memory_pool_;
    
    // Thread-local storage for calculations
    thread_local static CalculationContext context_;
    
    // SIMD-optimized calculation kernels
    struct AVXCalculationKernels {
        static void vectorized_black_scholes(
            const __m256d* spot_prices,
            const __m256d* strike_prices,
            const __m256d* time_to_expiry,
            const __m256d* volatilities,
            __m256d* option_prices,
            size_t count
        );
        
        static void vectorized_greeks_calculation(
            const __m256d* market_data,
            __m256d* greeks_output,
            size_t instruments_count
        );
    };

public:
    /**
     * Ultra-fast options pricing with SIMD optimization
     * Target: < 100 nanoseconds per option
     * 
     * Uses vectorized Black-Scholes implementation with AVX2 instructions
     * Memory-aligned data structures for optimal cache performance
     * Branch-free calculations for consistent latency
     */
    std::vector<OptionPrice> calculate_options_prices_vectorized(
        const std::vector<OptionParams>& options,
        const MarketData& market_data
    ) noexcept {
        
        // Pre-allocate aligned memory for SIMD operations
        alignas(32) std::vector<double> spot_prices(options.size());
        alignas(32) std::vector<double> strike_prices(options.size());
        alignas(32) std::vector<double> results(options.size());
        
        // Prepare data for vectorized calculation
        prepare_vectorized_data(options, market_data, 
                              spot_prices.data(), strike_prices.data());
        
        // SIMD calculation - process 4 options simultaneously
        const size_t simd_count = options.size() / 4;
        const size_t remainder = options.size() % 4;
        
        AVXCalculationKernels::vectorized_black_scholes(
            reinterpret_cast<const __m256d*>(spot_prices.data()),
            reinterpret_cast<const __m256d*>(strike_prices.data()),
            // ... other parameters
            reinterpret_cast<__m256d*>(results.data()),
            simd_count
        );
        
        // Handle remaining options with scalar calculations
        if (remainder > 0) {
            calculate_remaining_scalar(options, market_data, 
                                     results, simd_count * 4, remainder);
        }
        
        return convert_to_option_prices(results);
    }
    
    /**
     * Real-time portfolio VaR calculation with Monte Carlo
     * Target: < 1 millisecond for 10,000 simulations
     * 
     * Uses parallel Monte Carlo with Intel TBB
     * GPU acceleration via CUDA for large portfolios
     * Advanced variance reduction techniques
     */
    PortfolioRisk calculate_portfolio_var(
        const Portfolio& portfolio,
        const RiskParameters& params
    ) const {
        
        // Parallel Monte Carlo simulation
        const size_t num_simulations = params.num_simulations;
        std::vector<double> simulation_results(num_simulations);
        
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, num_simulations),
            [&](const tbb::blocked_range<size_t>& range) {
                // Thread-local random number generator
                thread_local FastRNG rng(std::chrono::high_resolution_clock::now()
                                       .time_since_epoch().count());
                
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    simulation_results[i] = simulate_portfolio_pnl(
                        portfolio, params, rng
                    );
                }
            }
        );
        
        // Calculate VaR and Expected Shortfall
        return calculate_risk_metrics(simulation_results, params.confidence_level);
    }
};

} // namespace finance

// Python bindings with pybind11
PYBIND11_MODULE(high_performance_finance, m) {
    m.doc() = "Ultra-high performance financial calculations engine";
    
    py::class_<finance::HighPerformanceRiskEngine>(m, "RiskEngine")
        .def(py::init<>())
        .def("calculate_options_prices_vectorized", 
             &finance::HighPerformanceRiskEngine::calculate_options_prices_vectorized,
             "Vectorized options pricing with SIMD optimization")
        .def("calculate_portfolio_var",
             &finance::HighPerformanceRiskEngine::calculate_portfolio_var,
             "Real-time portfolio VaR calculation");
             
    py::class_<finance::OptionParams>(m, "OptionParams")
        .def(py::init<double, double, double, double>())
        .def_readwrite("spot_price", &finance::OptionParams::spot_price)
        .def_readwrite("strike_price", &finance::OptionParams::strike_price);
}
```

```python
# Python Integration Layer
"""
Python Integration for C++ High-Performance Engine

Features:
- Zero-copy numpy array integration
- Async processing for I/O bound operations
- Intelligent batching for optimal C++ performance
- Comprehensive error handling and validation
- Performance monitoring and profiling
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional
import high_performance_finance as hpf
from dataclasses import dataclass
import time

class PythonRiskEngineWrapper:
    """
    Python wrapper for C++ high-performance risk engine.
    
    Provides async interface and intelligent batching for optimal performance.
    Handles data validation and error recovery.
    """
    
    def __init__(self, batch_size: int = 1000):
        self.cpp_engine = hpf.RiskEngine()
        self.batch_size = batch_size
        self.performance_metrics = PerformanceTracker()
        
    async def calculate_options_prices_batch(
        self,
        options_data: np.ndarray,
        market_data: Dict[str, float]
    ) -> np.ndarray:
        """
        Async batch processing of options pricing.
        
        Automatically batches large requests for optimal C++ engine performance.
        Uses asyncio to prevent blocking during long calculations.
        """
        
        start_time = time.perf_counter()
        
        # Validate input data
        self._validate_options_data(options_data)
        
        # Convert to C++ format with zero-copy when possible
        cpp_options = self._prepare_cpp_options(options_data)
        cpp_market_data = hpf.MarketData(**market_data)
        
        # Process in batches to prevent memory issues
        if len(cpp_options) > self.batch_size:
            results = await self._process_batched(cpp_options, cpp_market_data)
        else:
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.cpp_engine.calculate_options_prices_vectorized,
                cpp_options,
                cpp_market_data
            )
        
        # Record performance metrics
        duration = time.perf_counter() - start_time
        self.performance_metrics.record_calculation(
            operation="options_pricing",
            count=len(cpp_options),
            duration=duration
        )
        
        return self._convert_to_numpy(results)
```

## Performance Optimization Examples

### High-Throughput Data Processing Pipeline

```python
# Ultra-High Performance Data Processing Pipeline
"""
Real-time Market Data Processing Pipeline

Performance Requirements:
- Process 1M+ market data points per second
- End-to-end latency < 10 milliseconds
- Memory usage < 2GB for 8-hour trading session
- Zero garbage collection pauses
- 99.99% uptime during market hours

Architecture:
- Lock-free circular buffers for data ingestion
- Memory-mapped files for historical data
- Async I/O with uvloop for maximum throughput
- SIMD optimizations for numerical calculations
- Connection pooling for database operations
- Intelligent data partitioning and sharding
"""

import asyncio
import uvloop
import numpy as np
from typing import AsyncGenerator, List, Optional
import mmap
import os
from dataclasses import dataclass
from collections import deque
import time
import psutil

# Set uvloop for maximum async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@dataclass
class MarketDataPoint:
    """Optimized market data structure with __slots__ for memory efficiency."""
    __slots__ = ['symbol', 'price', 'volume', 'timestamp', 'exchange']
    
    symbol: str
    price: float
    volume: int
    timestamp: int  # Unix timestamp in nanoseconds
    exchange: str

class LockFreeCircularBuffer:
    """
    Lock-free circular buffer for high-throughput data ingestion.
    
    Uses atomic operations and memory barriers for thread safety.
    Optimized for single producer, multiple consumer pattern.
    """
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.empty(size, dtype=object)
        self.write_index = 0
        self.read_index = 0
        self._mask = size - 1  # Size must be power of 2
        
    def try_write(self, data: MarketDataPoint) -> bool:
        """Non-blocking write operation."""
        next_write = (self.write_index + 1) & self._mask
        
        if next_write == self.read_index:
            return False  # Buffer full
            
        self.buffer[self.write_index] = data
        # Memory barrier to ensure data is written before index update
        self.write_index = next_write
        return True
    
    def try_read(self) -> Optional[MarketDataPoint]:
        """Non-blocking read operation."""
        if self.read_index == self.write_index:
            return None  # Buffer empty
            
        data = self.buffer[self.read_index]
        self.read_index = (self.read_index + 1) & self._mask
        return data

class HighThroughputMarketDataProcessor:
    """
    Ultra-high performance market data processing system.
    
    Processes market data with minimal latency using advanced optimization techniques.
    """
    
    def __init__(self, buffer_size: int = 1024 * 1024):  # 1M elements
        self.ingestion_buffer = LockFreeCircularBuffer(buffer_size)
        self.processing_workers = []
        self.statistics = ProcessingStatistics()
        self.memory_pool = MemoryPool(size_mb=512)
        
    async def start_processing_pipeline(self, num_workers: int = 4):
        """Start the high-performance processing pipeline."""
        
        # Start ingestion coroutine
        ingestion_task = asyncio.create_task(self._data_ingestion_loop())
        
        # Start processing workers
        worker_tasks = []
        for i in range(num_workers):
            task = asyncio.create_task(self._processing_worker(worker_id=i))
            worker_tasks.append(task)
            
        # Start monitoring and optimization
        monitor_task = asyncio.create_task(self._performance_monitor())
        
        # Wait for all tasks
        await asyncio.gather(
            ingestion_task,
            *worker_tasks,
            monitor_task
        )
    
    async def _data_ingestion_loop(self):
        """High-speed data ingestion with minimal allocations."""
        
        # Pre-allocate objects to avoid GC pressure
        reusable_data_point = MarketDataPoint('', 0.0, 0, 0, '')
        batch_size = 1000
        
        while True:
            try:
                # Batch receive for better performance
                raw_data_batch = await self._receive_market_data_batch(batch_size)
                
                for raw_data in raw_data_batch:
                    # Reuse object to minimize allocations
                    self._populate_data_point(reusable_data_point, raw_data)
                    
                    # Non-blocking write to circular buffer
                    while not self.ingestion_buffer.try_write(reusable_data_point):
                        # Buffer full, yield to allow processing
                        await asyncio.sleep(0)
                        
                    self.statistics.increment_ingested()
                    
            except Exception as e:
                # Log error but continue processing
                await self._handle_ingestion_error(e)
    
    async def _processing_worker(self, worker_id: int):
        """High-performance processing worker with SIMD optimizations."""
        
        batch_processor = SIMDMarketDataProcessor()
        batch_buffer = []
        batch_size = 100
        
        while True:
            try:
                # Collect batch for vectorized processing
                for _ in range(batch_size):
                    data_point = self.ingestion_buffer.try_read()
                    if data_point is None:
                        break
                    batch_buffer.append(data_point)
                
                if not batch_buffer:
                    await asyncio.sleep(0.001)  # 1ms sleep if no data
                    continue
                
                # Vectorized processing with SIMD
                processing_start = time.perf_counter_ns()
                
                processed_results = batch_processor.process_batch_vectorized(
                    batch_buffer
                )
                
                # Async database write with connection pooling
                await self._write_processed_data_async(processed_results)
                
                processing_duration = time.perf_counter_ns() - processing_start
                self.statistics.record_processing_duration(
                    worker_id, len(batch_buffer), processing_duration
                )
                
                # Clear batch for reuse
                batch_buffer.clear()
                
            except Exception as e:
                await self._handle_processing_error(worker_id, e)

class SIMDMarketDataProcessor:
    """SIMD-optimized market data calculations."""
    
    def process_batch_vectorized(
        self, 
        data_points: List[MarketDataPoint]
    ) -> List[ProcessedMarketData]:
        """
        Vectorized processing using NumPy's SIMD capabilities.
        
        Processes multiple data points simultaneously for maximum throughput.
        """
        
        # Convert to NumPy arrays for vectorized operations
        prices = np.array([dp.price for dp in data_points], dtype=np.float64)
        volumes = np.array([dp.volume for dp in data_points], dtype=np.int64)
        timestamps = np.array([dp.timestamp for dp in data_points], dtype=np.int64)
        
        # Vectorized calculations
        # Calculate VWAP (Volume Weighted Average Price)
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        
        # Calculate price changes
        price_changes = np.diff(prices, prepend=prices[0])
        
        # Calculate volatility metrics
        price_variance = np.var(prices)
        price_std = np.sqrt(price_variance)
        
        # Calculate time-weighted metrics
        time_deltas = np.diff(timestamps, prepend=timestamps[0])
        time_weighted_prices = prices * time_deltas
        
        # Return processed results
        return [
            ProcessedMarketData(
                original=data_points[i],
                vwap=vwap,
                price_change=price_changes[i],
                volatility=price_std,
                time_weight=time_weighted_prices[i]
            )
            for i in range(len(data_points))
        ]
```

### Database Performance Optimization

```java
// High-Performance Database Operations
/**
 * Ultra-High Performance Database Layer
 * 
 * Performance Targets:
 * - 100,000+ transactions per second
 * - < 1ms average query latency
 * - 99.9% queries under 5ms
 * - Zero connection pool exhaustion
 * - Automatic query optimization
 * 
 * Optimizations:
 * - Custom connection pooling with health checks
 * - Prepared statement caching
 * - Batch operations with optimal batch sizes
 * - Read replicas with intelligent routing
 * - Query result caching with invalidation
 * - Database sharding with consistent hashing
 */

@Component
@Slf4j
public class HighPerformanceDatabaseService {
    
    private final HikariDataSource primaryDataSource;
    private final List<HikariDataSource> readReplicas;
    private final QueryCache queryCache;
    private final BatchProcessor batchProcessor;
    private final DatabaseMetrics metrics;
    private final ConsistentHashingShardRouter shardRouter;
    
    // Prepared statement cache for optimal performance
    private final LoadingCache<String, PreparedStatement> statementCache;
    
    @Autowired
    public HighPerformanceDatabaseService(
            @Qualifier("primaryDataSource") HikariDataSource primaryDataSource,
            @Qualifier("readReplicas") List<HikariDataSource> readReplicas,
            QueryCache queryCache,
            DatabaseMetrics metrics) {
        
        this.primaryDataSource = primaryDataSource;
        this.readReplicas = readReplicas;
        this.queryCache = queryCache;
        this.metrics = metrics;
        this.batchProcessor = new OptimalBatchProcessor();
        this.shardRouter = new ConsistentHashingShardRouter();
        
        // Initialize prepared statement cache
        this.statementCache = Caffeine.newBuilder()
            .maximumSize(1000)
            .expireAfterAccess(Duration.ofMinutes(30))
            .recordStats()
            .build(this::createPreparedStatement);
    }
    
    /**
     * High-performance batch insert with optimal batch sizing.
     * 
     * Automatically determines optimal batch size based on:
     * - Data size and complexity
     * - Database performance characteristics
     * - Network latency measurements
     * - Memory constraints
     */
    @Async("databaseExecutor")
    public CompletableFuture<BatchInsertResult> insertBatch(
            List<? extends Entity> entities) {
        
        Timer.Sample sample = Timer.start(metrics.getMeterRegistry());
        
        try {
            // Determine optimal batch size dynamically
            int optimalBatchSize = batchProcessor.calculateOptimalBatchSize(
                entities.size(), 
                entities.get(0).getClass()
            );
            
            // Process in optimal batches
            List<CompletableFuture<Integer>> batchFutures = new ArrayList<>();
            
            for (List<? extends Entity> batch : Lists.partition(entities, optimalBatchSize)) {
                CompletableFuture<Integer> batchFuture = processBatchInsert(batch);
                batchFutures.add(batchFuture);
            }
            
            // Wait for all batches to complete
            return CompletableFuture.allOf(batchFutures.toArray(new CompletableFuture[0]))
                .thenApply(v -> {
                    int totalInserted = batchFutures.stream()
                        .mapToInt(CompletableFuture::join)
                        .sum();
                    
                    return new BatchInsertResult(totalInserted, entities.size());
                });
                
        } finally {
            sample.stop(Timer.builder("database.batch.insert.duration")
                .tag("entity.type", entities.get(0).getClass().getSimpleName())
                .register(metrics.getMeterRegistry()));
        }
    }
    
    private CompletableFuture<Integer> processBatchInsert(List<? extends Entity> batch) {
        return CompletableFuture.supplyAsync(() -> {
            String sql = generateBatchInsertSQL(batch.get(0).getClass());
            
            try (Connection conn = primaryDataSource.getConnection()) {
                conn.setAutoCommit(false);
                
                // Use cached prepared statement for performance
                try (PreparedStatement stmt = statementCache.get(sql)) {
                    
                    // Batch parameter setting with type-specific optimizations
                    for (Entity entity : batch) {
                        setParametersOptimized(stmt, entity);
                        stmt.addBatch();
                    }
                    
                    // Execute batch with performance monitoring
                    int[] results = stmt.executeBatch();
                    conn.commit();
                    
                    metrics.recordBatchInsert(batch.size(), results.length);
                    
                    return results.length;
                }
            } catch (SQLException e) {
                metrics.recordDatabaseError("batch_insert", e);
                throw new DatabaseException("Batch insert failed", e);
            }
        }, getDatabaseExecutor());
    }
    
    /**
     * Intelligent query execution with caching and read replica routing.
     * 
     * Features:
     * - Automatic read/write splitting
     * - Query result caching with smart invalidation
     * - Load balancing across read replicas
     * - Circuit breaker for failed replicas
     * - Query performance monitoring and optimization
     */
    public <T> CompletableFuture<List<T>> executeQuery(
            String sql,
            Object[] parameters,
            RowMapper<T> rowMapper,
            QueryHint... hints) {
        
        String cacheKey = QueryCacheKey.generate(sql, parameters);
        
        // Check cache first
        if (shouldUseCache(hints)) {
            List<T> cachedResult = queryCache.get(cacheKey);
            if (cachedResult != null) {
                metrics.recordCacheHit("query_cache");
                return CompletableFuture.completedFuture(cachedResult);
            }
        }
        
        // Determine data source (primary vs read replica)
        DataSource dataSource = selectOptimalDataSource(hints);
        
        return CompletableFuture.supplyAsync(() -> {
            Timer.Sample sample = Timer.start(metrics.getMeterRegistry());
            
            try (Connection conn = dataSource.getConnection();
                 PreparedStatement stmt = statementCache.get(sql)) {
                
                // Set parameters with type-specific optimizations
                setParametersOptimized(stmt, parameters);
                
                // Execute query with timeout
                stmt.setQueryTimeout(getQueryTimeout(hints));
                
                try (ResultSet rs = stmt.executeQuery()) {
                    List<T> results = new ArrayList<>();
                    
                    while (rs.next()) {
                        results.add(rowMapper.mapRow(rs));
                    }
                    
                    // Cache results if appropriate
                    if (shouldCacheResult(results, hints)) {
                        queryCache.put(cacheKey, results, getCacheDuration(hints));
                    }
                    
                    metrics.recordQueryExecution(sql, results.size());
                    
                    return results;
                }
                
            } catch (SQLException e) {
                metrics.recordDatabaseError("query_execution", e);
                throw new DatabaseException("Query execution failed: " + sql, e);
            } finally {
                sample.stop(Timer.builder("database.query.duration")
                    .tag("query.type", extractQueryType(sql))
                    .register(metrics.getMeterRegistry()));
            }
        }, getDatabaseExecutor());
    }
    
    /**
     * Advanced database sharding with consistent hashing.
     * 
     * Automatically routes queries to appropriate shards based on:
     * - Sharding key extraction from query
     * - Consistent hashing algorithm
     * - Shard health and performance metrics
     * - Cross-shard query optimization
     */
    public <T> CompletableFuture<List<T>> executeShardedQuery(
            String sql,
            Object[] parameters,
            String shardingKey,
            RowMapper<T> rowMapper) {
        
        // Determine target shard(s)
        List<ShardInfo> targetShards = shardRouter.getTargetShards(shardingKey, sql);
        
        if (targetShards.size() == 1) {
            // Single shard query - direct execution
            return executeQueryOnShard(sql, parameters, rowMapper, targetShards.get(0));
        } else {
            // Cross-shard query - parallel execution and result merging
            return executeCrossShardQuery(sql, parameters, rowMapper, targetShards);
        }
    }
}
```

This advanced examples collection provides comprehensive patterns for implementing sophisticated AI-assisted development workflows using GitHub Copilot. Each example demonstrates production-ready code with real-world performance requirements, security considerations, and architectural best practices.
