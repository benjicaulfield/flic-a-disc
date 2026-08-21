# API Error Training Data Generator

Generates multi-terminal debugging scenarios for training models on programming domain reasoning.

## Overview

This generator creates realistic debugging scenarios by:
1. Intentionally introducing type mismatches between services
2. Starting Go backend and Django ML service
3. Sending API requests that trigger errors
4. Capturing logs from all terminals
5. Saving structured training data

## Usage

```bash
# Generate all scenarios
python generate_api_errors.py

# Generate only first 3 scenarios
python generate_api_errors.py --count 3

# Custom output directory
python generate_api_errors.py --output ./my_training_data
```

## Output Structure

```
training_data/
├── summary.json                    # Overview of all scenarios
├── 001_discogs_id_int_vs_string/
│   ├── metadata.json              # Scenario details
│   ├── go_backend.txt             # Go server logs
│   ├── ml_service.txt             # Django ML logs
│   ├── curl_request.txt           # Request and response
│   └── model_code.txt             # Code snippet with issue
├── 002_budget_string_vs_float/
│   └── ...
└── 003_missing_required_field/
    └── ...
```

## Scenarios Included

### 1. `discogs_id_int_vs_string` (Easy)
- **Error**: Frontend sends string, Go expects int
- **Terminals**: 4 (Go, ML, curl, code)
- **Learning**: Type system mismatches across languages

### 2. `budget_string_vs_float` (Easy)
- **Error**: Decimal number vs integer type
- **Terminals**: 3 (Go, curl, code)
- **Learning**: Numeric type precision

### 3. `missing_required_field` (Easy)
- **Error**: Missing required field in request
- **Terminals**: 2 (Go, curl)
- **Learning**: API contract validation

### 4. `array_vs_object` (Medium)
- **Error**: Wrong JSON structure
- **Terminals**: 2 (Go, curl)
- **Learning**: JSON schema validation

### 5. `null_vs_empty_string` (Medium)
- **Error**: Null vs empty string handling
- **Terminals**: 3 (Go, curl, code)
- **Learning**: Nullable types and pointers

## Example Output

### `go_backend.txt`
```
[GIN] 2026/03/25 - 20:15:32 | 500 | 123.45ms | ::1 | POST "/api/discogs/knapsack"
Decode error: json: cannot unmarshal string into Go struct field KnapsackItem.discogs_id of type int
```

### `curl_request.txt`
```bash
$ curl -X POST \
  http://localhost:8000/api/discogs/knapsack \
  -H "Content-Type: application/json" \
  -d '{
  "seller": "test_seller",
  "budget": 200
}'

Response Status: 500
Response Body:
{"error": "Failed to parse ML response"}
```

### `model_code.txt`
```go
  95 |
  96 | type KnapsackItem struct {
  97 |     DiscogsID      int            `json:"discogs_id"`  // ← Modified line
  98 |     Artist         string         `json:"artist"`
  99 |     Title          string         `json:"title"`
```

## Training Tasks

These scenarios can be used for:

1. **Error localization**: "Which terminal shows the root cause?"
2. **Error explanation**: "Explain what's happening in plain English"
3. **Fix suggestion**: "What code change would fix this?"
4. **Causality**: "Which terminal output caused which other output?"
5. **Multi-hop reasoning**: "Trace the request through all services"

## Extending

Add new scenarios to the `scenarios()` method:

```python
{
    "name": "your_scenario_name",
    "description": "What goes wrong",
    "file": "path/to/file.go",  # or None
    "original": "original line",
    "modified": "modified line",
    "request": {
        "url": "http://localhost:8000/api/endpoint",
        "method": "POST",
        "json": {"your": "data"}
    },
    "expected_error": "error message",
    "difficulty": "easy|medium|hard",
    "terminals": ["list", "of", "terminals"]
}
```

## Requirements

- Go backend running
- Django ML service available
- PostgreSQL database running
- Python 3.11+

## Safety

- Automatically backs up files before modification
- Restores original code after each scenario
- Rebuilds Go binary after modifications
- Cleans up temporary files
